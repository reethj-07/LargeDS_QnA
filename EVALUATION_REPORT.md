# Evaluation Report

## Methodology

- **Ground truth:** For each test question, `ground_truth_ids` lists `reviews.id` values from DuckDB queries that approximate *relevant* documents (e.g. keyword filter, category filter, same ASIN). This is a **strict** SQL view of relevance.

- **Retrieval:** Hybrid search (FAISS + BM25 + RRF, then optional **cross-encoder** rerank on a pool of candidates) uses `retrieval_query` per question — a short keyword-style string (see `evaluation/test_questions.json`). Full natural-language questions were found to dilute BM25; short queries align better with SQL-derived labels.

- **Metrics:** **Recall@K** = |retrieved ∩ relevant| / |relevant|; **Precision@K** = |retrieved ∩ relevant| / K; **Hit@K** = 1 if at least one relevant id appears in the top-K results; **MRR@K** = reciprocal rank of the first relevant hit; **nDCG@K** = normalized discounted cumulative gain (binary relevance).

- **Interpretation:** Absolute scores are often modest because SQL relevance sets are large and diverse while the retriever returns only 10 dense+sparse-fused candidates. Use metrics for **before/after** comparisons (e.g. ablations), not as an absolute ceiling.


## Retrieval (hybrid FAISS + BM25 + RRF + optional cross-encoder)

| id | tier | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Hit@10 |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | simple | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 2 | simple | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 3 | simple | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 4 | simple | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 5 | aggregation | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 6 | aggregation | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 7 | aggregation | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 8 | aggregation | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 9 | comparison | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 10 | comparison | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 11 | comparison | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 12 | comparison | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 13 | multi_hop | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 14 | multi_hop | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 15 | multi_hop | 0.08333333333333333 | 0.1 | 1.0 | 1.0 | 1 |
| 16 | multi_hop | 0.1 | 0.1 | 0.2 | 0.38685280723454163 | 1 |
| 17 | trend | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 18 | trend | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 19 | trend | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| 20 | trend | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

### Aggregate (n=20)

- **Mean Recall@10:** 0.0092
- **Mean Precision@10:** 0.0100
- **Mean MRR@10:** 0.0600
- **Mean nDCG@10:** 0.0693
- **Hit@10 rate:** 2/20 questions with ≥1 overlap in top-10

### Mean Recall@10 by tier

- **aggregation:** 0.0000 (n=4)
- **comparison:** 0.0000 (n=4)
- **multi_hop:** 0.0458 (n=4)
- **simple:** 0.0000 (n=4)
- **trend:** 0.0000 (n=4)

## Answer quality (qualitative)

- The **Analyst** (Groq `llama-3.3-70b-versatile`) synthesizes answers with SQL + retrieved context; the **Critic** scores grounding 1–5. Spot-check complex queries via `python scripts/query.py "..."`.

- **Example (manual):** Aggregation query *average rating by category* — SQL returned per-category averages; a follow-up tie case (All_Beauty vs Amazon_Fashion) initially mis-stated “the” lowest category until prompts required explicit **ties**; after prompt adjustment, critic score reached 5/5.


## Failure cases and error analysis

1. **Planner-generated SQL:** Occasionally invalid or overly narrow SQL is skipped; retrieval falls back to hybrid search only (`retrieval_sql_error` in logs).

2. **Numeric ties:** Category averages can tie at displayed precision; the analyst must list all tied groups (mitigated in `src/agents/analyst.py`).

3. **Retrieval vs SQL labels:** Low Recall@10 does not always mean poor UX — users may still get correct **aggregates** from SQL while retrieved snippets illustrate themes.

4. **Groq rate limits:** Heavy evaluation or rapid UI clicks can hit limits; add backoff or cache for production.


## Trade-offs

- **CPU embeddings** (~1h+ for 150k rows) vs cloud GPU — chosen for zero cost and reproducibility.

- **MiniLM (384-d)** vs larger encoders — smaller index and RAM footprint on a 12GB laptop; trade some semantic nuance.

- **RRF + cross-encoder** — RRF fuses dense+sparse lists; a small MS MARCO cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks a pool on CPU. Set `USE_CROSS_ENCODER=0` to skip rerank for speed or CI.

- **Groq free tier** — fast and free; vendor lock-in and rate limits vs self-hosted LLM.


## Ablation study

| Mode | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | Hit rate |
|------|---:|---:|---:|---:|---:|
| Vector only | 0.0042 | 0.0050 | 0.0167 | 0.0250 | 0.0500 |
| BM25 only | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Hybrid (RRF) | 0.0042 | 0.0050 | 0.0100 | 0.0193 | 0.0500 |
| Hybrid + CE | 0.0092 | 0.0100 | 0.0600 | 0.0693 | 0.1000 |

## Answer quality (quantitative, batch)

- **Questions scored:** 19/20
- **Mean critic score:** 4.58 / 5
- **Min:** 3.0 | **Max:** 5.0
- **Score >= 4 (good/great):** 17/19
- **Mean latency:** 25.4s per question
- **Pipeline errors:** 1/20

## Regenerating this report

```powershell
python scripts/build_ground_truth.py   # after ingest; optional
python scripts/evaluate.py
python scripts/ablation.py             # optional ablation
python scripts/answer_quality.py       # optional batch answer quality
```
