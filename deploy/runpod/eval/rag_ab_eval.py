"""A/B of evidence retrieval for guidance turns, as locked in ab_khai_sinh.json.

Stage 1, collect: inside the image venv, on the stack network. Read-only access to
Postgres and Qdrant. Every variant goes through the ai-service's own run_retrieve.
Writes raw.json, which holds chunk excerpts: keep it out of Git.

  python rag_ab_eval.py collect --spec production_khai_sinh.json --ab ab_khai_sinh.json \
    --requests production_requests_khai_sinh.json --fp32-dir DIR [--int8-dir DIR \
    --simulate-vnni-flags] --out raw.json
  Env: DATABASE_URL, QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_VARIANT=fp32

Stage 2, report: on the host, needs Go. The displayed citations are chosen by the
production Go code (apps/api/internal/chat/display_eval_test.go). Writes results
without chunk text.

  python3 rag_ab_eval.py report --spec ... --ab ... --requests ... --raw raw.json \
    --out results.json [--fail-on-gate]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["A_baseline", "B_topk20", "C_topk24", "D_short_query", "E_per_statement"]
QUESTION_VARIANTS = ["A_baseline", "B_topk20", "C_topk24"]


def sha_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clip_query(text: str, limit: int = 2000) -> str:
    # Go retrievalQuery: strings.Fields join, then clipRunes(2000).
    return " ".join(text.split())[:limit]


# ---------------------------------------------------------------- collect ----
def collect(args) -> None:
    sys.path.insert(0, str(HERE))
    import e5_production_eval as pe
    from app.indexing.embed import OnnxEmbedder
    from app.indexing.retrieve import EmbedderIdentity, http_search, postgres_loader, run_retrieve
    from app.models.retrieve import MAX_TOP_K, RetrieveRequest

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    ab = json.loads(Path(args.ab).read_text(encoding="utf-8"))
    requests = json.loads(Path(args.requests).read_text(encoding="utf-8"))["requests"]
    sc = spec["scope"]
    dsn = pe.read_only_dsn(os.environ["DATABASE_URL"])
    qdrant, coll = os.environ["QDRANT_URL"], os.environ.get("QDRANT_COLLECTION", "knowledge_chunks")
    rows = pe.load_rows(dsn, sc["generation_id"])
    points = pe.scroll_points(qdrant, coll, sc["generation_id"])
    by_id = {str(r["id"]): r for r in rows}
    labels = pe.topic_chunks(spec, rows)
    name, code = ab["variants"]["D_short_query"]["procedure_name"], ab["variants"]["D_short_query"]["procedure_code"]

    def one(embedder, search, load, query, statements, top_k):
        identity = EmbedderIdentity(model_id=embedder.model_id, revision=embedder.revision, checksum=str(embedder.checksum))
        req = RetrieveRequest(schema_version="retrieve.v1", query=query, xa_id=sc["xa_id"],
                              procedure_version_id=sc["procedure_version_id"],
                              allowed_sources=[{"document_id": sc["document_id"], "generation_id": sc["generation_id"]}],
                              top_k=min(top_k, MAX_TOP_K), statements=statements)
        if top_k > MAX_TOP_K:
            # The /v1/retrieve contract refuses top_k > MAX_TOP_K (20). Measured anyway,
            # flagged in the results as needing a contract change.
            req.top_k = top_k
        return run_retrieve(req, embed_query=embedder.embed_query, search=search, load_chunks=load, embed_identity=identity)

    def variant_case(variant, embedder, search, load, query, statements):
        """One turn of a variant -> status, reason, wire hits, query sent to Go, calls."""
        if variant in ("A_baseline", "B_topk20", "C_topk24"):
            k = {"A_baseline": 12, "B_topk20": 20, "C_topk24": 24}[variant]
            resp = one(embedder, search, load, query, statements, k)
            return resp.status, resp.reason, [h.model_dump(mode="json") for h in resp.hits], query, 1
        if variant == "D_short_query":
            short = clip_query(f"{name} {code} " + " ".join(statements))
            resp = one(embedder, search, load, short, statements, 12)
            return resp.status, resp.reason, [h.model_dump(mode="json") for h in resp.hits], short, 1
        # E: one search per statement, round-robin merge in checklist order, dedup by chunk_id.
        subs = [one(embedder, search, load, clip_query(f"{name} {s}"), statements, 12) for s in statements]
        lists = [[h.model_dump(mode="json") for h in r.hits] if r.status == "ok" else [] for r in subs]
        merged, seen = [], set()
        for rank in range(max((len(x) for x in lists), default=0)):
            for lst in lists:
                if rank < len(lst) and lst[rank]["chunk_id"] not in seen:
                    seen.add(lst[rank]["chunk_id"])
                    merged.append(lst[rank])
        merged = merged[:12]
        status, reason = ("ok", None) if merged else (subs[0].status, subs[0].reason)
        dup_removed = sum(len(x) for x in lists) - len(seen)
        return status, reason, merged, query, len(statements), dup_removed

    cases = [("request", r["id"], r["query"], r["statements"]) for r in requests]
    cases += [("question", f"Q{i + 1}", q["q"], []) for i, q in enumerate(spec["citizen_questions"])]

    def run_all(run_name, embedder, search, load, timed):
        out = {}
        for variant in VARIANTS:
            for kind, cid, query, statements in cases:
                if kind == "question" and variant not in QUESTION_VARIANTS:
                    continue
                res = variant_case(variant, embedder, search, load, query, statements)
                ms = []
                if timed:  # warm-up done by the call above; median of 3 timed turns
                    for _ in range(3):
                        t = time.perf_counter()
                        variant_case(variant, embedder, search, load, query, statements)
                        ms.append((time.perf_counter() - t) * 1000)
                out[f"{variant}|{cid}"] = {
                    "variant": variant, "case": cid, "kind": kind, "status": res[0], "reason": res[1],
                    "hits": res[2], "go_query": res[3], "statements": statements,
                    "calls": {"embed": res[4], "search": res[4], "postgres": res[4]},
                    "dedup_removed": res[5] if len(res) > 5 else 0,
                    "ms": round(statistics.median(ms), 1) if ms else None,
                }
        return {"run": run_name, "variant": embedder.variant, "checksum": embedder.checksum, "cases": out}

    fp32 = OnnxEmbedder(args.fp32_dir)
    if fp32.variant != "fp32":
        sys.exit("EMBEDDING_VARIANT must select fp32 for --fp32-dir")
    runs = {"live-fp32": run_all("live-fp32", fp32, http_search(qdrant, coll, 10), postgres_loader(dsn, 10), True)}
    if args.int8_dir:
        os.environ["EMBEDDING_VARIANT"] = "int8-vnni"
        if args.simulate_vnni_flags:
            os.environ["EMBEDDING_CPU_FLAGS"] = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni"
        int8 = OnnxEmbedder(args.int8_dir)
        os.environ.pop("EMBEDDING_CPU_FLAGS", None)
        texts = [str(r["content"]) for r in rows]
        vec8 = dict(zip(by_id, int8.embed_batch(texts)))
        rows8 = {i: dict(r, embedding_checksum=int8.checksum) for i, r in by_id.items()}
        runs["offline-int8"] = run_all("offline-int8", int8, pe.memory_search(points, vec8),
                                       lambda ids: [rows8[i] for i in ids if i in rows8], False)
    corpus_fp = hashlib.sha256("\n".join(f"{r['id']}:{r['text_sha256']}" for r in rows).encode()).hexdigest()
    raw = {
        "provenance": {
            "cas_revision": os.environ.get("CAS_REVISION", ""), "spec_sha256": sha_file(args.spec),
            "ab_spec_sha256": sha_file(args.ab), "requests_sha256": sha_file(args.requests),
            "corpus_fingerprint_sha256": corpus_fp, "generation_id": sc["generation_id"], "chunks": len(rows),
            "onnxruntime": __import__("onnxruntime").__version__,
            "int8_cpu_flags_simulated": bool(args.int8_dir and args.simulate_vnni_flags),
            "latency_note": "live-fp32 only: median of 3 turns after one warm-up; dev Qdrant/Postgres on the same Mac",
            "contract_max_top_k": MAX_TOP_K,
            "C_topk24_note": "top_k=24 exceeds the /v1/retrieve contract (MAX_TOP_K); measured by setting top_k after validation",
        },
        "labels": labels, "runs": runs,
    }
    Path(args.out).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    print(f"collected {sum(len(r['cases']) for r in runs.values())} turns into {args.out}")


# ----------------------------------------------------------------- report ----
def go_display(raw: dict, spec: dict, repo: Path) -> dict:
    sc = spec["scope"]
    cases = []
    for run_name, run in raw["runs"].items():
        for key, c in run["cases"].items():
            if c["kind"] != "request":
                continue
            cases.append({"id": f"{run_name}|{key}", "query": c["go_query"], "statements": c["statements"],
                          "status": c["status"], "reason": c["reason"] or "", "hits": c["hits"]})
    harness_in = {"xa_id": sc["xa_id"], "procedure_id": spec["scope"].get("procedure_id", ""),
                  "procedure_version_id": sc["procedure_version_id"], "procedure_code": sc["procedure_code"],
                  "pins": [{"document_id": sc["document_id"], "generation_id": sc["generation_id"]}], "cases": cases}
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp, "in.json"), Path(tmp, "out.json")
        fin.write_text(json.dumps(harness_in, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ, CAS_DISPLAY_EVAL_IN=str(fin), CAS_DISPLAY_EVAL_OUT=str(fout))
        subprocess.run(["go", "test", "./internal/chat", "-run", "^TestDisplayEvalHarness$", "-count=1"],
                       cwd=repo / "apps/api", env=env, check=True, capture_output=True, text=True)
        return {r["id"]: r for r in json.loads(fout.read_text(encoding="utf-8"))}


def report(args) -> None:
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    ab = json.loads(Path(args.ab).read_text(encoding="utf-8"))
    requests = json.loads(Path(args.requests).read_text(encoding="utf-8"))["requests"]
    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    if raw["provenance"]["ab_spec_sha256"] != sha_file(args.ab):
        sys.exit("ab spec changed since collect")
    repo = HERE.parents[2]
    if not spec["scope"].get("procedure_id"):
        sys.exit("scope.procedure_id missing in spec")
    shown = go_display(raw, spec, repo)
    labels = {t: set(v) for t, v in raw["labels"].items()}

    rules = spec["labels"]["statement_topics"]

    def topics_of(statement):
        for rule in rules:
            if rule["contains"].lower() in statement.lower():
                return rule["topics"]
        sys.exit(f"statement has no label rule: {statement[:80]}")

    def covered(ids, topics):
        want = set().union(*(labels[t] for t in topics))
        return any(i in want for i in ids)

    # Equivalence of the Go harness with what users saw (live fp32 baseline).
    gen_sha = {}
    for c in raw["runs"]["live-fp32"]["cases"].values():
        for h in c["hits"]:
            gen_sha[h["content_sha256"]] = h["chunk_id"]
    equivalence = []
    for r in requests:
        rec = r["recorded_display_order_content_sha256"]
        got = [x["content_sha256"] for x in shown[f"live-fp32|A_baseline|{r['id']}"]["citations"] or []]
        comparable = all(s in gen_sha for s in rec)
        equivalence.append({"request": r["id"], "comparable": comparable, "same_order": got == rec if comparable else None,
                            "recorded": [s[:12] for s in rec], "harness": [s[:12] for s in got]})
    comp = [e for e in equivalence if e["comparable"]]
    exact = bool(comp) and all(e["same_order"] for e in comp)

    out_runs, metrics = {}, {}
    for run_name, run in raw["runs"].items():
        metrics[run_name], out_runs[run_name] = {}, {}
        for variant in VARIANTS:
            cs = {k.split("|", 1)[1]: c for k, c in run["cases"].items() if c["variant"] == variant}
            st12, stk, std, complete, ms, searches, dups, ncit, ok_req, dedup = [], [], [], [], [], [], 0, 0, 0, 0
            for r in requests:
                c = cs[r["id"]]
                ids = [h["chunk_id"] for h in c["hits"]]
                disp = shown[f"{run_name}|{variant}|{r['id']}"]
                did = [x["chunk_id"] for x in disp["citations"] or []]
                tps = [topics_of(s) for s in r["statements"]]
                st12 += [covered(ids[:12], t) for t in tps]
                stk += [covered(ids, t) for t in tps]
                dcov = [covered(did, t) for t in tps]
                std += dcov
                complete.append(all(dcov))
                ok_req += int(disp["retrieval_status"] == "ok" and len(did) > 0)
                seen_sha, seen_page = set(), set()
                for x in disp["citations"] or []:
                    ncit += 1
                    if x["content_sha256"] in seen_sha or x["page_start"] in seen_page:
                        dups += 1
                    seen_sha.add(x["content_sha256"])
                    seen_page.add(x["page_start"])
                searches.append(c["calls"]["search"])
                dedup += c["dedup_removed"]
                if c["ms"] is not None:
                    ms.append(c["ms"])
            m = {
                "Coverage@12": round(sum(st12) / len(st12), 3), "Coverage@k": round(sum(stk) / len(stk), 3),
                "DisplayCoverage@3": round(sum(std) / len(std), 3),
                "RequestComplete@display": round(sum(complete) / len(complete), 3),
                "requests_ok_with_citation": f"{ok_req}/{len(requests)}",
                "searches_per_turn_max": max(searches), "searches_per_turn_mean": round(sum(searches) / len(searches), 2),
                "displayed_citations": ncit, "display_duplication": round(dups / max(1, ncit), 3),
                "per_statement_duplicates_removed": dedup,
                "statement_turns": len(st12),
            }
            if ms:
                q = statistics.quantiles(ms, n=20) if len(ms) > 1 else [ms[0]] * 19
                m["latency_ms_p50"], m["latency_ms_p95"] = round(statistics.median(ms), 1), round(q[18], 1)
            if variant in QUESTION_VARIANTS:
                hit, compl, tc, tt, rr = [], [], 0, 0, []
                for i, qq in enumerate(spec["citizen_questions"]):
                    ids = [h["chunk_id"] for h in cs[f"Q{i + 1}"]["hits"]]
                    got = [bool(set(ids[:5]) & labels[t]) for t in qq["topics"]]
                    want = set().union(*(labels[t] for t in qq["topics"]))
                    ranks = [n + 1 for n, h in enumerate(ids[:10]) if h in want]
                    hit.append(bool(ranks and ranks[0] <= 5))
                    compl.append(all(got))
                    tc, tt = tc + sum(got), tt + len(got)
                    rr.append(1 / ranks[0] if ranks else 0.0)
                m.update({"Q_Hit@5": round(sum(hit) / len(hit), 3), "Q_Complete@5": round(sum(compl) / len(compl), 3),
                          "Q_TopicCoverage@5": round(tc / tt, 3), "Q_MRR@10": round(sum(rr) / len(rr), 3)})
            metrics[run_name][variant] = m
            out_runs[run_name][variant] = {
                cid: {"status": c["status"], "reason": c["reason"], "calls": c["calls"], "ms": c["ms"],
                      "hits": [{"chunk_id": h["chunk_id"], "page": h["page_start"], "score": round(h["retrieval_score"], 4)} for h in c["hits"]],
                      "displayed": [{"chunk_id": x["chunk_id"], "page": x["page_start"], "evidence_status": x["evidence_status"]}
                                    for x in (shown.get(f"{run_name}|{variant}|{cid}") or {}).get("citations") or []]}
                for cid, c in cs.items()}

    # Decision rule, as locked in ab_khai_sinh.json.
    A, A8 = metrics["live-fp32"]["A_baseline"], metrics.get("offline-int8", {}).get("A_baseline")
    decisions = {}
    for v in VARIANTS[1:]:
        m, m8 = metrics["live-fp32"][v], metrics.get("offline-int8", {}).get(v)
        checks = {
            "display_coverage_plus_0.10": m["DisplayCoverage@3"] >= A["DisplayCoverage@3"] + 0.10,
            "coverage12_not_lower": m["Coverage@12"] >= A["Coverage@12"],
            "all_requests_ok_with_citation": m["requests_ok_with_citation"] == f"{len(requests)}/{len(requests)}",
            "questions_not_lower": (m["Q_Hit@5"] >= A["Q_Hit@5"] and m["Q_Complete@5"] >= A["Q_Complete@5"]) if "Q_Hit@5" in m else None,
            "latency_p95_ok": m["latency_ms_p95"] <= 2 * A["latency_ms_p95"] and m["latency_ms_p95"] <= 1000,
            "searches_per_turn_le_8": m["searches_per_turn_max"] <= 8,
            "duplication_ok": m["display_duplication"] <= A["display_duplication"] + 0.10,
            "int8_same_direction": (m8["DisplayCoverage@3"] >= A8["DisplayCoverage@3"]) if m8 and A8 else None,
        }
        decisions[v] = {"pass": all(x is not False for x in checks.values()), "checks": checks}
    passing = [v for v in decisions if decisions[v]["pass"]]
    passing.sort(key=lambda v: (metrics["live-fp32"][v]["searches_per_turn_max"], metrics["live-fp32"][v]["latency_ms_p95"]))
    result = {
        "provenance": dict(raw["provenance"], ab_spec_sha256_at_report=sha_file(args.ab),
                           harness_sha256=sha_file(repo / "apps/api/internal/chat/display_eval_test.go"),
                           go_version=subprocess.run(["go", "version"], capture_output=True, text=True).stdout.strip()),
        "display_equivalence": {"exact_on_comparable": exact, "comparable": len(comp), "cases": equivalence,
                                "display_metric": "exact (production Go code)" + ("" if exact else ", NOT reproduced: treat as approximate")},
        "metrics": metrics, "decisions": decisions,
        "recommended_by_rule": passing[0] if passing else None,
        "scope_of_claim": ab["decision_rule"]["scope_of_claim"],
        "runs": out_runs,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("display_equivalence", "metrics", "decisions", "recommended_by_rule")}, ensure_ascii=False, indent=1))
    if args.fail_on_gate and not exact:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "report"):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
        p.add_argument("--ab", required=True)
        p.add_argument("--requests", required=True)
        p.add_argument("--out", required=True)
    c = sub.choices["collect"]
    c.add_argument("--fp32-dir", required=True)
    c.add_argument("--int8-dir")
    c.add_argument("--simulate-vnni-flags", action="store_true")
    r = sub.choices["report"]
    r.add_argument("--raw", required=True)
    r.add_argument("--fail-on-gate", action="store_true", help="exit 1 when the Go display harness does not reproduce the recorded citations")
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else report(args)


if __name__ == "__main__":
    main()
