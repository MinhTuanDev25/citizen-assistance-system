"""Retrieval check on the production chat path, per E5 variant.

Calls the ai-service's own run_retrieve (Qdrant filter on xa_id + procedure_version_id +
document/generation, relevance floor, query overlap, Postgres row checks) with the
request shape Go sends (top_k=12, statements = Definition checklist). Read-only:
Qdrant search/scroll and SELECTs in a READ ONLY Postgres session.

Runs:
  live-fp32      real Qdrant vectors (written by the indexer) + real Postgres rows
  offline-<v>    the same payloads and rows; passage vectors re-embedded with variant
                 <v> (the generation as if it had been indexed with <v>)

  python e5_production_eval.py --spec production_khai_sinh.json \
    --requests production_requests_khai_sinh.json --fp32-dir DIR [--int8-dir DIR \
    --simulate-vnni-flags] --out results.json
Env: DATABASE_URL (read-only use), QDRANT_URL, QDRANT_COLLECTION, CAS_REVISION (optional).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import unicodedata
from pathlib import Path

import httpx

from app.indexing.embed import OnnxEmbedder
from app.indexing.retrieve import EmbedderIdentity, http_search, postgres_loader, run_retrieve
from app.models.retrieve import RetrieveRequest


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[,;:()/\-]", " ", text)
    return re.sub(r"\s+", " ", text)


def sha_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_only_dsn(dsn: str) -> str:
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}options=-c%20default_transaction_read_only%3Don"


def load_rows(dsn: str, generation_id: str) -> list[dict]:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT id::text FROM knowledge_chunks WHERE generation_id = %s ORDER BY page_start, id", (generation_id,))
        ids = [row[0] for row in cur.fetchall()]
    rows = postgres_loader(dsn, 10)(ids)
    order = {cid: i for i, cid in enumerate(ids)}
    return sorted(rows, key=lambda r: order[str(r["id"])])


def scroll_points(base: str, collection: str, generation_id: str) -> list[dict]:
    points, offset = [], None
    while True:
        body = {"filter": {"must": [{"key": "generation_id", "match": {"value": generation_id}}]},
                "limit": 256, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        r = httpx.post(f"{base.rstrip('/')}/collections/{collection}/points/scroll", json=body, timeout=30)
        r.raise_for_status()
        result = r.json()["result"]
        points += result["points"]
        offset = result.get("next_page_offset")
        if offset is None:
            return points


def match_filter(flt: dict, payload: dict) -> bool:
    for cond in flt.get("must", []):
        if "should" in cond:
            if not any(match_filter(sub, payload) for sub in cond["should"]):
                return False
        elif str(payload.get(cond["key"])) != str(cond["match"]["value"]):
            return False
    return True


def memory_search(points: list[dict], vectors: dict[str, list[float]]):
    def search(vector, flt, limit):
        hits = []
        for p in points:
            if match_filter(flt, p["payload"]):
                v = vectors[p["payload"]["chunk_id"]]
                hits.append({"id": p["id"], "score": float(sum(a * b for a, b in zip(vector, v))), "payload": p["payload"]})
        hits.sort(key=lambda h: -h["score"])
        return hits[:limit]

    return search


def topic_chunks(spec: dict, rows: list[dict]) -> dict[str, list[str]]:
    out = {}
    for topic, phrases in spec["labels"]["topics"].items():
        out[topic] = [str(r["id"]) for r in rows if any(p in norm(str(r["content"])) for p in phrases)]
        if not out[topic]:
            sys.exit(f"label topic {topic} matches no chunk")
    return out


def statement_topics(spec: dict, statement: str) -> list[str]:
    for rule in spec["labels"]["statement_topics"]:
        if rule["contains"].lower() in statement.lower():
            return rule["topics"]
    sys.exit(f"statement has no label rule: {statement[:80]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--fp32-dir", required=True)
    ap.add_argument("--int8-dir")
    ap.add_argument("--simulate-vnni-flags", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail-on-gate", action="store_true", help="exit 1 when a threshold fails (CI quality gate)")
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    requests = json.loads(Path(args.requests).read_text(encoding="utf-8"))["requests"]
    sc = spec["scope"]
    dsn = read_only_dsn(os.environ["DATABASE_URL"])
    qdrant, coll = os.environ["QDRANT_URL"], os.environ.get("QDRANT_COLLECTION", "knowledge_chunks")

    rows = load_rows(dsn, sc["generation_id"])
    points = scroll_points(qdrant, coll, sc["generation_id"])
    if len(rows) != sc["generation_chunk_count"] or len(points) != len(rows):
        sys.exit(f"generation size mismatch: rows={len(rows)} points={len(points)}")
    by_id = {str(r["id"]): r for r in rows}
    labels = topic_chunks(spec, rows)
    corpus_fp = hashlib.sha256("\n".join(f"{r['id']}:{r['text_sha256']}" for r in rows).encode()).hexdigest()

    def make_request(query: str, statements: list[str]) -> RetrieveRequest:
        return RetrieveRequest(
            schema_version="retrieve.v1", query=query, xa_id=sc["xa_id"],
            procedure_version_id=sc["procedure_version_id"],
            allowed_sources=[{"document_id": sc["document_id"], "generation_id": sc["generation_id"]}],
            top_k=sc["top_k"], statements=statements,
        )

    cases = [("request", r["id"], r["query"], r["statements"]) for r in requests]
    cases += [("question", f"Q{i + 1}", q["q"], []) for i, q in enumerate(spec["citizen_questions"])]

    def run(name: str, embedder: OnnxEmbedder, search, load) -> dict:
        identity = EmbedderIdentity(model_id=embedder.model_id, revision=embedder.revision, checksum=str(embedder.checksum))
        out = {}
        for kind, cid, query, statements in cases:
            t = time.perf_counter()
            resp = run_retrieve(make_request(query, statements), embed_query=embedder.embed_query,
                                search=search, load_chunks=load, embed_identity=identity)
            out[cid] = {"status": resp.status, "reason": resp.reason, "ms": round((time.perf_counter() - t) * 1000, 1),
                        "hits": [h.chunk_id for h in resp.hits], "pages": [h.page_start for h in resp.hits],
                        "scores": [round(h.retrieval_score, 4) for h in resp.hits]}
        return {"name": name, "variant": embedder.variant, "checksum": embedder.checksum, "cases": out}

    runs = {}
    fp32 = OnnxEmbedder(args.fp32_dir)
    if fp32.variant != "fp32":
        sys.exit("EMBEDDING_VARIANT must select fp32 for --fp32-dir")
    runs["live-fp32"] = run("live-fp32", fp32, http_search(qdrant, coll, 10), postgres_loader(dsn, 10))
    texts = [str(r["content"]) for r in rows]
    vec32 = dict(zip(by_id, fp32.embed_batch(texts)))
    stored = {p["payload"]["chunk_id"]: p["vector"] for p in points}
    cos_stored = [sum(a * b for a, b in zip(stored[c], vec32[c])) for c in by_id]
    runs["offline-fp32"] = run("offline-fp32", fp32, memory_search(points, vec32), lambda ids: [by_id[i] for i in ids if i in by_id])
    if args.int8_dir:
        os.environ["EMBEDDING_VARIANT"] = "int8-vnni"
        if args.simulate_vnni_flags:
            os.environ["EMBEDDING_CPU_FLAGS"] = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni"
        int8 = OnnxEmbedder(args.int8_dir)
        os.environ.pop("EMBEDDING_CPU_FLAGS", None)
        vec8 = dict(zip(by_id, int8.embed_batch(texts)))
        rows8 = {i: dict(r, embedding_checksum=int8.checksum) for i, r in by_id.items()}
        runs["offline-int8"] = run("offline-int8", int8, memory_search(points, vec8), lambda ids: [rows8[i] for i in ids if i in rows8])

    # ---- metrics ------------------------------------------------------------------
    def covered(hits, topics, k):
        want = set().union(*(labels[t] for t in topics))
        return any(h in want for h in hits[:k])

    def request_metrics(run_):
        per, flat12, flat3 = {}, [], []
        for r in requests:
            c = run_["cases"][r["id"]]
            st = [statement_topics(spec, s) for s in r["statements"]]
            c12 = [covered(c["hits"], t, sc["top_k"]) for t in st]
            c3 = [covered(c["hits"], t, sc["display_k"]) for t in st]
            flat12 += c12
            flat3 += c3
            per[r["id"]] = {"status": c["status"], "coverage12": round(sum(c12) / len(c12), 3), "score_top3_coverage": round(sum(c3) / len(c3), 3)}
        return {"per_request": per, "statement_coverage12": round(sum(flat12) / len(flat12), 3),
                "statement_score_top3_coverage": round(sum(flat3) / len(flat3), 3), "statements": len(flat12),
                "status_ok": sum(1 for v in per.values() if v["status"] == "ok")}

    def question_metrics(run_):
        # hit5: any labelled topic in the top 5 (named recall5 before 2026-10-30, same formula).
        # complete5: every labelled topic in the top 5. topic_coverage5: topics covered / topics.
        hit, comp, rr, tc, tt, per = [], [], [], 0, 0, {}
        for i, q in enumerate(spec["citizen_questions"]):
            c = run_["cases"][f"Q{i + 1}"]
            want = set().union(*(labels[t] for t in q["topics"]))
            ranks = [n + 1 for n, h in enumerate(c["hits"][:10]) if h in want]
            top5 = set(c["hits"][:5])
            got = [bool(top5 & set(labels[t])) for t in q["topics"]]
            hit.append(bool(ranks and ranks[0] <= 5))
            comp.append(all(got))
            tc, tt = tc + sum(got), tt + len(got)
            rr.append(1 / ranks[0] if ranks else 0.0)
            per[f"Q{i + 1}"] = {"status": c["status"], "reason": c["reason"], "first_relevant_rank": ranks[0] if ranks else None,
                                "topics_in_top5": [t for t, g in zip(q["topics"], got) if g], "topics_missing_top5": [t for t, g in zip(q["topics"], got) if not g]}
        return {"hit5": round(sum(hit) / len(hit), 3), "complete5": round(sum(comp) / len(comp), 3),
                "topic_coverage5": round(tc / tt, 3), "topics": tt, "mrr10": round(sum(rr) / len(rr), 3), "per_question": per}

    metrics = {name: {"requests": request_metrics(r), "questions": question_metrics(r)} for name, r in runs.items()}

    # A recorded citation is comparable only when its text exists in this generation
    # (older turns may cite an older generation with other chunk boundaries).
    sha_of = {str(r["text_sha256"]): str(r["id"]) for r in rows}
    cite_total, cite_in, cite_other = 0, 0, 0
    cite_detail = []
    for r in requests:
        live = runs["live-fp32"]["cases"][r["id"]]
        for c in r["recorded_citations"]:
            cid = sha_of.get(c["content_sha256"])
            if cid is None:
                cite_other += 1
                cite_detail.append({"request": r["id"], "result": "text_not_in_generation"})
                continue
            cite_total += 1
            found = cid in live["hits"][: sc["top_k"]]
            cite_in += int(found)
            score_now = live["scores"][live["hits"].index(cid)] if found else None
            cite_detail.append({"request": r["id"], "result": "in_top12" if found else "missing",
                                "recorded_score": round(c["retrieval_score"], 4), "live_score": score_now})

    same = [runs["live-fp32"]["cases"][k]["hits"] == runs["offline-fp32"]["cases"][k]["hits"] for k in runs["live-fp32"]["cases"]]
    pairs = {}
    if "offline-int8" in runs:
        a, b = runs["offline-fp32"]["cases"], runs["offline-int8"]["cases"]
        pairs["offline-fp32 vs offline-int8"] = {
            "top12_overlap_mean": round(sum(len(set(a[k]["hits"]) & set(b[k]["hits"])) / max(1, len(set(a[k]["hits"]) | set(b[k]["hits"]))) for k in a) / len(a), 3),
            "top3_same_set_share": round(sum(set(a[k]["hits"][:3]) == set(b[k]["hits"][:3]) for k in a) / len(a), 3),
            "status_same_share": round(sum(a[k]["status"] == b[k]["status"] for k in a) / len(a), 3),
        }

    m32, live = metrics["offline-fp32"], metrics["live-fp32"]
    m8 = metrics.get("offline-int8")
    th = {
        "P1_live_status": (live["requests"]["status_ok"] == len(requests), f"{live['requests']['status_ok']}/{len(requests)} ok"),
        "P2_harness_fidelity": (sum(same) / len(same) >= 0.95, f"{sum(same)}/{len(same)} identical hit lists"),
        "P3_statement_coverage": (live["requests"]["statement_coverage12"] >= 0.80, f"coverage@12 {live['requests']['statement_coverage12']}"),
        # P5 denominator was changed after the first run (see the spec); both results are reported.
        "P5_recorded_citations": (cite_total > 0 and cite_in / cite_total >= 0.90,
                                  f"adjusted after first run: {cite_in}/{cite_total} comparable citations in live top-12; "
                                  f"{cite_other} cite text of another generation, not evaluated. Original definition: "
                                  f"{cite_in}/{cite_total + cite_other} = {'PASS' if cite_in / max(1, cite_total + cite_other) >= 0.90 else 'FAIL'}"),
        "Q1_questions_hit5": (live["questions"]["hit5"] >= 0.80,
                              f"Hit@5 {live['questions']['hit5']} (any labelled topic; Complete@5 {live['questions']['complete5']}, "
                              f"TopicCoverage@5 {live['questions']['topic_coverage5']})"),
    }
    if m8:
        th["P4_variant_coverage"] = (m32["requests"]["statement_coverage12"] >= m8["requests"]["statement_coverage12"] - 0.05,
                                     f"fp32 {m32['requests']['statement_coverage12']} vs int8 {m8['requests']['statement_coverage12']}")
        th["Q2_questions_variant"] = (m32["questions"]["hit5"] >= m8["questions"]["hit5"] - 0.05 and m32["questions"]["mrr10"] >= m8["questions"]["mrr10"] - 0.03,
                                      f"Hit@5 {m32['questions']['hit5']} vs {m8['questions']['hit5']}, MRR@10 {m32['questions']['mrr10']} vs {m8['questions']['mrr10']}")
    report = {
        "provenance": {
            "cas_revision": os.environ.get("CAS_REVISION", ""),
            "spec_sha256": sha_file(args.spec), "requests_sha256": sha_file(args.requests),
            "corpus_fingerprint_sha256": corpus_fp, "corpus_fingerprint_rule": "sha256 of '<chunk id>:<text_sha256>' lines, rows ordered by page_start, id",
            "generation_id": sc["generation_id"], "chunks": len(rows),
            "models": {name: {"variant": r["variant"], "checksum": r["checksum"]} for name, r in runs.items()},
            "int8_cpu_flags_simulated": bool(args.int8_dir and args.simulate_vnni_flags),
            "machine": platform.machine(), "onnxruntime": __import__("onnxruntime").__version__,
            "stored_vs_reembedded_fp32_cosine_min": round(min(cos_stored), 6),
        },
        "thresholds": {k: {"pass": bool(v[0]), "evidence": v[1]} for k, v in th.items()},
        "metrics": metrics, "pairs": pairs, "recorded_citations": cite_detail,
        "missed_statements": {name: [{"request": r["id"], "topics": statement_topics(spec, s), "statement": s[:120]}
                                     for r in requests for s in r["statements"]
                                     if not covered(run_["cases"][r["id"]]["hits"], statement_topics(spec, s), sc["top_k"])]
                              for name, run_ in runs.items()},
        "labels": {t: [{"chunk_id": c, "page": by_id[c]["page_start"]} for c in ids] for t, ids in labels.items()},
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"thresholds": report["thresholds"], "pairs": pairs,
                      "summary": {n: {"coverage12": m["requests"]["statement_coverage12"], "score_top3_coverage": m["requests"]["statement_score_top3_coverage"],
                                      "q_hit5": m["questions"]["hit5"], "q_complete5": m["questions"]["complete5"],
                                      "q_topic_coverage5": m["questions"]["topic_coverage5"], "q_mrr10": m["questions"]["mrr10"]} for n, m in metrics.items()},
                      "provenance": report["provenance"]}, ensure_ascii=False, indent=1))
    failed = [k for k, v in th.items() if not v[0]]
    if args.fail_on_gate and failed:
        print(f"GATE FAIL: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
