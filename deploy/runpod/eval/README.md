# E5 and evidence retrieval checks

All tools here are read-only: Qdrant search/scroll, and SELECTs in a READ ONLY Postgres session. They do not change Go retrieval, the ai-service runtime, Qdrant, E5 weights, ASR or MT.

| File | Role |
|---|---|
| `production_khai_sinh.json` | Scope (generation, version), chunk-level silver labels, citizen questions, thresholds, and the history of metric corrections |
| `production_requests_khai_sinh.json` | The 7 distinct `retrieval_query` values Go really sent, with checklist statements, recorded citations and their display order |
| `e5_production_eval.py` | Production chat path per E5 variant. `--fail-on-gate` exits 1 when a threshold fails |
| `ab_khai_sinh.json` | **A/B spec, locked before the first A/B run** (SHA-256 `0ac74e28…`): variants, merge rule, metrics, decision rule |
| `rag_ab_eval.py` | A/B runner: `collect` runs in the image; `report` runs on the host and uses Go |
| `../../../apps/api/internal/chat/display_eval_test.go` | Test-only harness. It runs the **production Go display path** (`attachHTTPSources`: filters, lexical overlap, `classifySupport`, `selectDisplayCitations`) on supplied hits and is skipped in normal test runs |
| `e5_variant_eval.py`, `queries_hotich_v2.json` | Whole-document search with no filter. Reference only; production never searches this way |
| `results/*.json` | Results with provenance: spec, requests and A/B spec SHA-256, corpus fingerprint, models, onnxruntime, harness SHA-256, Go version. They also hold per-turn hits (chunk id, page, score), the displayed citations and the labels. **No chunk text** |

## How production retrieves evidence

On guidance turns, Go builds the E5 query from the pinned Definition, not from the citizen's message (`apps/api/internal/chat/retrieve.go`, `retrievalQuery`). The query is the name, code, guidance summary and checklist. It calls `/v1/retrieve` with `top_k=12`, filters the hits, and shows up to 3 citations: supporting ones first, then in hit order.

The retrieval only looks for evidence for the Definition's guidance; the Decision Engine reply does not change. Whether citations should also answer the citizen's specific question is a product decision. It is not evaluated here.

## Reproduce

```bash
# 1. collect, inside the image venv on the stack network (read-only)
docker run --rm --network <stack network> --entrypoint /opt/cas/venv-ai/bin/python \
  -e DATABASE_URL=... -e QDRANT_URL=http://qdrant:6333 -e EMBEDDING_VARIANT=fp32 -e PYTHONPATH=/ai \
  -v $PWD/apps/ai-service:/ai:ro -v $PWD/deploy/runpod/eval:/ev:ro -v <fp32>:/m/fp32:ro -v <int8>:/m/int8:ro -v <scratch>:/s \
  <cas-aio image> /ev/rag_ab_eval.py collect --spec /ev/production_khai_sinh.json --ab /ev/ab_khai_sinh.json \
  --requests /ev/production_requests_khai_sinh.json --fp32-dir /m/fp32 --int8-dir /m/int8 --simulate-vnni-flags --out /s/ab_raw.json
# 2. report on the host (Go); ab_raw.json holds chunk excerpts, keep it out of Git
python3 deploy/runpod/eval/rag_ab_eval.py report --spec ... --ab ... --requests ... --raw <scratch>/ab_raw.json \
  --out deploy/runpod/eval/results/<name>.json --fail-on-gate
```

`e5_production_eval.py` runs the same way; see its docstring. Before comparing a re-run, check that `corpus_fingerprint_sha256` matches.

## Results of 2026-10-30

- Generation `d6872f43…`, 87 chunks.
- Corpus fingerprint `f764d32a…`.
- 7 production requests, 30 statement-turns (18 distinct statements).
- 15 citizen questions.
- Machine: arm64. int8 runs use simulated VNNI flags, so they are used for quality only, never for latency.

### Production path (`results/production_khai_sinh_2026-10-30.json`)

| Threshold | Result |
|---|---|
| P1 live status ok | PASS 7/7 |
| P2 offline fp32 = live fp32 | PASS 22/22 |
| P3 statement Coverage@12 ≥ 0.80 | **FAIL 0.767** (23/30). The script exits 1 with `--fail-on-gate` |
| P4 fp32 ≥ int8 − 0.05 | PASS (0.767 vs 0.733) |
| P5 recorded citations in live top-12 | Definition fixed before the run: 15/21 = **FAIL**. After the denominator correction made after the first run: 15/15 = PASS, with 6 citations of an older generation not evaluated. Both are reported; see `metric_corrections_after_first_run` |
| Q1 Hit@5 ≥ 0.80 | PASS 1.0. This was called Recall@5 before; the formula is unchanged. Also reported: Complete@5 0.933 (14/15), TopicCoverage@5 0.95 (19/20) |
| Q2 fp32 vs int8 on questions | PASS |

`statement_score_top3_coverage` (0.433) in this file takes the top 3 by score. It is **not** the displayed set. The exact displayed coverage is `DisplayCoverage@3` in the A/B results.

### A/B (`results/ab_khai_sinh_2026-10-30.json`)

The Go display harness reproduced the recorded citations, content and order, for **5/5** comparable requests (R3–R7). R1 and R2 cite an older generation and could not be compared. `DisplayCoverage@3` is therefore the production Go result, not an approximation.

**fp32 live (dev Qdrant/Postgres on the same Mac)**

| Variant | Coverage@12 | Coverage@k | DisplayCoverage@3 | Req. complete @display | Searches/turn | p50 / p95 ms | Display dup. | Q Hit@5 / Complete@5 |
|---|---|---|---|---|---|---|---|---|
| A baseline (top 12) | 0.767 | 0.767 | 0.433 | 0.143 | 1 | 60 / 73 | 0.19 | 1.0 / 0.933 |
| B top_k 20 | 0.767 | 0.900 | 0.433 | 0.143 | 1 | 59 / 84 | 0.19 | 1.0 / 0.933 |
| C top_k 24 ¹ | 0.767 | 1.000 | 0.433 | 0.143 | 1 | 58 / 74 | 0.19 | 1.0 / 0.933 |
| D short query | 0.800 | 0.800 | **0.533** | 0.286 | 1 | 52 / 68 | 0.143 | n/a |
| E per statement | **0.867** | 0.867 | 0.500 | 0.143 | up to 6 | 170 / 217 | 0.095 | n/a |

¹ `top_k=24` exceeds the `/v1/retrieve` contract (`MAX_TOP_K = 20`). It was measured by setting `top_k` after validation and would need a contract change.

**int8 offline (quality only)**

| Variant | DisplayCoverage@3 | Coverage@12 |
|---|---|---|
| A | 0.467 | 0.733 |
| B | 0.467 | 0.733 |
| C | 0.467 | 0.733 |
| D | 0.533 | 0.733 |
| E | 0.467 | 0.867 |

**Decision by the locked rule**

| Variant | Result | Failing checks |
|---|---|---|
| B | not passed | DisplayCoverage@3 unchanged |
| C | not passed | DisplayCoverage@3 unchanged |
| D | passed | — |
| E | not passed | DisplayCoverage@3 +0.067 < +0.10; p95 217 ms > 2 × 73 ms |

What the results mean:
- A larger `top_k` finds more evidence deeper in the list but never changes the 3 citations shown. Go puts supporting hits first and keeps hit order, and the extra hits rank lower.
- D passes **exactly at the threshold**: 16/30 vs 13/30, a difference of 3 statement-turns. That is a weak signal.
- D drops the guidance summary from the query, while the labels only measure the checklist. Evidence that the summary asked for, such as nơi nộp and trình tự, is not measured, so D may lose it.
- E raises Coverage@12 the most. It costs up to 6 embeds and 6 searches per turn, and that does not reach the 3 displayed citations.
- With 5–6 statements and only 3 citations, `RequestComplete@display` is capped well below 1 for every variant.

Limits: 7 requests from one procedure, silver labels that no domain expert has reviewed, and dev-Mac latency. **This does not show an improvement for every procedure.** Before any production change, test on an independent question set and review the shown citations by hand.

### Whole document, no filter (reference): `results/variant_fulldoc_2026-10-30.json`

- 3215 chunks, fingerprint `8ee2182d…`, 35 questions.
- fp32 Recall@5 0.857, MRR@10 0.629; int8 0.829 / 0.576.
- Procedure 1 Recall@5 is 0.556 for both, because the near-identical khai sinh procedures compete.
- int8 vs fp32 cosine: queries mean 0.997, passages mean 0.994 (min 0.982).
- fp32 on emulated x86_64 AVX2 equals arm64: cosine 1.00000, identical top-10 on the 87-chunk subset.
- Not measured: real AMD EPYC 7B13, and real VNNI int8 on the full corpus.
