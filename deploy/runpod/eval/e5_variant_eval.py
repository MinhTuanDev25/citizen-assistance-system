"""Compare E5 embedding variants (int8-vnni vs fp32) on a fixed corpus and question set.

Uses the ai-service's own OnnxEmbedder (same tokenizer, prefixes, pooling and
normalization as indexing and retrieval). Retrieval is exact cosine search over
all chunks, i.e. what Qdrant's cosine search approximates.

  # one process per variant and CPU (clean timing and RSS):
  python e5_variant_eval.py embed --variant fp32 --model-dir DIR --corpus corpus.jsonl \
      --queries queries_hotich_v2.json --out fp32_<cpu>.npz
  # metrics, thresholds and comparisons:
  python e5_variant_eval.py report --queries queries_hotich_v2.json --corpus corpus.jsonl \
      [--per-query] fp32_x.npz int8_x.npz fp32_y.npz ...

corpus.jsonl: one {"id", "page_start", "page_end", "content"} per line, e.g. exported
read-only from knowledge_chunks of one READY generation. The script never writes to
Postgres, MinIO or Qdrant.

--simulate-vnni-flags runs the int8 graph on a CPU without AVX-512 VNNI by setting the
test-only EMBEDDING_CPU_FLAGS override (e.g. arm64, whose int8 dot product is exact like
VNNI). The run is marked cpu_flags_simulated; it is never a production setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


def load_corpus(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def cpu_label() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def cmd_embed(args) -> None:
    os.environ["EMBEDDING_VARIANT"] = args.variant
    if args.simulate_vnni_flags:
        os.environ["EMBEDDING_CPU_FLAGS"] = "avx512f,avx512bw,avx512dq,avx512vl,avx512vnni"
    from app.indexing.embed import OnnxEmbedder, detect_cpu_flags

    corpus = load_corpus(args.corpus)
    spec = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    rss0 = rss_mb()
    t0 = time.perf_counter()
    emb = OnnxEmbedder(args.model_dir, batch_size=args.batch)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    passages = np.asarray(emb.embed_batch([row["content"] for row in corpus]), dtype=np.float32)
    passage_s = time.perf_counter() - t0
    q_lat, queries = [], []
    for item in spec["queries"]:
        emb.embed_query(item["q"])  # warm
    for item in spec["queries"]:
        t = time.perf_counter()
        queries.append(emb.embed_query(item["q"]))
        q_lat.append((time.perf_counter() - t) * 1000)
    os.environ.pop("EMBEDDING_CPU_FLAGS", None)
    flags = detect_cpu_flags()
    meta = {
        "variant_requested": args.variant,
        "variant": emb.variant,
        "checksum": emb.checksum,
        "machine": platform.machine(),
        "cpu": cpu_label(),
        "avx2": "avx2" in flags,
        "avx512_vnni": "avx512vnni" in flags,
        "cpu_flags_simulated": bool(args.simulate_vnni_flags),
        "onnxruntime": __import__("onnxruntime").__version__,
        "load_s": round(load_s, 2),
        "passages": len(corpus),
        "passage_embed_s": round(passage_s, 1),
        "passage_ms_per_chunk": round(passage_s * 1000 / max(1, len(corpus)), 2),
        "query_ms_p50": round(float(np.percentile(q_lat, 50)), 1),
        "query_ms_p95": round(float(np.percentile(q_lat, 95)), 1),
        "rss_before_mb": round(rss0, 0),
        "rss_peak_mb": round(rss_mb(), 0),
    }
    np.savez(args.out, passages=passages, queries=np.asarray(queries, dtype=np.float32),
             ids=np.asarray([row["id"] for row in corpus]), meta=json.dumps(meta))
    print(json.dumps(meta, ensure_ascii=False))


def _relevant(spec: dict, corpus: list[dict], target: str, family: bool) -> np.ndarray:
    procs = spec["procedures"]
    wanted = [procs[target]]
    if family:
        wanted = [p for p in procs.values() if p["family"] == procs[target]["family"]]
    rel = np.zeros(len(corpus), dtype=bool)
    for i, row in enumerate(corpus):
        a, b = int(row["page_start"] or 0), int(row["page_end"] or row["page_start"] or 0)
        rel[i] = any(a <= p["pages"][1] and b >= p["pages"][0] for p in wanted)
    return rel


def _metrics(run: dict, spec: dict, corpus: list[dict]) -> dict:
    scores = run["queries"] @ run["passages"].T
    order = np.argsort(-scores, axis=1, kind="stable")
    out = {"per_query": []}
    for qi, item in enumerate(spec["queries"]):
        row = {"q": item["q"], "group": item["group"], "top10": order[qi, :10].tolist()}
        for label, fam in (("strict", False), ("family", True)):
            rel = _relevant(spec, corpus, item["p"], fam)
            hits = rel[order[qi, :10]]
            first = int(np.argmax(hits)) + 1 if hits.any() else 0
            row[f"hit5_{label}"] = bool(hits[:5].any())
            row[f"rr10_{label}"] = 1.0 / first if first else 0.0
            row[f"rank_{label}"] = first or None
        out["per_query"].append(row)

    def agg(rows):
        return {
            "n": len(rows),
            "recall5_strict": round(float(np.mean([r["hit5_strict"] for r in rows])), 3),
            "mrr10_strict": round(float(np.mean([r["rr10_strict"] for r in rows])), 3),
            "recall5_family": round(float(np.mean([r["hit5_family"] for r in rows])), 3),
            "mrr10_family": round(float(np.mean([r["rr10_family"] for r in rows])), 3),
        }

    rows = out["per_query"]
    out["all"] = agg(rows)
    for group in sorted({r["group"] for r in rows}):
        out[group] = agg([r for r in rows if r["group"] == group])
    return out


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


def cmd_report(args) -> None:
    corpus = load_corpus(args.corpus)
    spec = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    runs = {}
    for path in args.runs:
        data = np.load(path)
        if list(data["ids"]) != [row["id"] for row in corpus]:
            sys.exit(f"{path}: corpus ids differ from {args.corpus}")
        runs[Path(path).stem] = {"passages": data["passages"], "queries": data["queries"], "meta": json.loads(str(data["meta"]))}
    fingerprint = hashlib.sha256("\n".join(
        f"{row['id']}:{hashlib.sha256(row['content'].encode('utf-8')).hexdigest()}" for row in corpus).encode()).hexdigest()
    report = {"provenance": {"corpus_rows": len(corpus), "corpus_fingerprint_sha256": fingerprint,
                             "corpus_fingerprint_rule": "sha256 of '<chunk id>:<sha256(content)>' lines in corpus order",
                             "queries_sha256": hashlib.sha256(Path(args.queries).read_bytes()).hexdigest()},
              "runs": {}, "pairs": {}}
    for name, run in runs.items():
        m = _metrics(run, spec, corpus)
        run["metrics"] = m
        report["runs"][name] = {"meta": run["meta"], **{k: v for k, v in m.items() if k != "per_query"}}
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ra, rb = runs[a], runs[b]
            cq = _cos_rows(ra["queries"], rb["queries"])
            cp = _cos_rows(ra["passages"], rb["passages"])
            ta = [r["top10"] for r in ra["metrics"]["per_query"]]
            tb = [r["top10"] for r in rb["metrics"]["per_query"]]
            report["pairs"][f"{a} vs {b}"] = {
                "cos_query_mean": round(float(cq.mean()), 5), "cos_query_min": round(float(cq.min()), 5),
                "cos_passage_mean": round(float(cp.mean()), 5), "cos_passage_min": round(float(cp.min()), 5),
                "cos_passage_p01": round(float(np.percentile(cp, 1)), 5),
                "top5_overlap_mean": round(float(np.mean([len(set(x[:5]) & set(y[:5])) / 5 for x, y in zip(ta, tb)])), 3),
                "top10_identical_share": round(float(np.mean([x == y for x, y in zip(ta, tb)])), 3),
                "top1_same_share": round(float(np.mean([x[0] == y[0] for x, y in zip(ta, tb)])), 3),
            }
    if args.per_query:
        ids = [row["id"] for row in corpus]
        report["per_query"] = {
            name: [{"q": r["q"], "rank_strict": r["rank_strict"], "rank_family": r["rank_family"],
                    "top5_chunk_ids": [ids[j] for j in r["top10"][:5]],
                    "top5_pages": [corpus[j]["page_start"] for j in r["top10"][:5]]} for r in run["metrics"]["per_query"]]
            for name, run in runs.items()
        }
    print(json.dumps(report, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--variant", required=True, choices=["fp32", "int8-vnni"])
    e.add_argument("--model-dir", required=True)
    e.add_argument("--corpus", required=True)
    e.add_argument("--queries", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--simulate-vnni-flags", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--corpus", required=True)
    r.add_argument("--queries", required=True)
    r.add_argument("--per-query", action="store_true")
    r.add_argument("runs", nargs="+")
    args = ap.parse_args()
    cmd_embed(args) if args.cmd == "embed" else cmd_report(args)


if __name__ == "__main__":
    main()
