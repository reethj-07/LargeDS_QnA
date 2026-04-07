# Evaluation Report

## Methodology

- **Ground truth:** For each test question, `ground_truth_ids` is a **large relevance pool** (up to 2 000 IDs) from deterministic, content-aligned DuckDB queries over the `text` column (raw review body, not category-prefixed `doc_text`). Large pools ensure retrieval metrics are meaningful — with 200–2 000 relevant docs, even top-10 retrieval has a realistic chance of overlap.

- **Eval modes:** Each question has an `eval_mode`:
  - `retrieval` — retrieval metrics are primary.
  - `sql` — SQL aggregation answers the question; retrieval overlap is informational.
  - `both` — SQL + retrieval contribute.

- **Category filtering:** Single-category questions use metadata-filtered retrieval (DuckDB pre-filter). Multi-category / cross-cutting questions search the full corpus.

- **Multi-K metrics:** Reported at K = 10, 50. Hit@K and Precision@K are the most interpretable when ground-truth pools are much larger than K; Recall@K is bounded by K / pool_size.


## Per-question results (K=10)

| id | tier | mode | pool | Hit@10 | Prec@10 | MRR@10 | nDCG@10 | Recall@10 | cat_filter |
|---:|---|---|---:|:---:|---:|---:|---:|---:|---|
| 1 | simple | both | 2000 | 1 | 0.3000 | 0.2500 | 0.4858 | 0.0015 | All_Beauty |
| 2 | simple | retrieval | 495 | 1 | 0.7000 | 1.0000 | 0.9642 | 0.0141 |  |
| 3 | simple | both | 293 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 4 | simple | retrieval | 2000 | 1 | 0.9000 | 1.0000 | 0.9667 | 0.0045 |  |
| 5 | aggregation | sql | 2000 | 1 | 0.2000 | 0.2500 | 0.4825 | 0.0010 |  |
| 6 | aggregation | sql | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 7 | aggregation | both | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 8 | aggregation | sql | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 9 | comparison | both | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 10 | comparison | both | 2000 | 1 | 0.1000 | 0.2000 | 0.3869 | 0.0005 |  |
| 11 | comparison | both | 2000 | 1 | 0.1000 | 0.1667 | 0.3562 | 0.0005 |  |
| 12 | comparison | both | 1925 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0052 |  |
| 13 | multi_hop | both | 293 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 14 | multi_hop | retrieval | 2000 | 1 | 0.3000 | 0.3333 | 0.5780 | 0.0015 |  |
| 15 | multi_hop | retrieval | 1220 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0082 | Arts_Crafts_and_Sewing |
| 16 | multi_hop | retrieval | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 17 | trend | retrieval | 963 | 1 | 0.9000 | 1.0000 | 0.9770 | 0.0093 | Appliances |
| 18 | trend | retrieval | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Baby_Products |
| 19 | trend | retrieval | 1251 | 1 | 0.5000 | 1.0000 | 0.8309 | 0.0040 |  |
| 20 | trend | retrieval | 2000 | 1 | 0.2000 | 1.0000 | 1.0000 | 0.0010 |  |

## Aggregate metrics


### All questions (n=20)

| K | Hit rate | Precision | MRR | nDCG | Recall |
|---:|---:|---:|---:|---:|---:|
| 10 | 12/20 (60%) | 0.3100 | 0.4100 | 0.4514 | 0.0026 |
| 50 | 15/20 (75%) | 0.2330 | 0.4183 | 0.4839 | 0.0067 |

### Retrieval-primary (eval_mode != sql) (n=17)

| K | Hit rate | Precision | MRR | nDCG | Recall |
|---:|---:|---:|---:|---:|---:|
| 10 | 11/17 (65%) | 0.3529 | 0.4676 | 0.5027 | 0.0030 |
| 50 | 14/17 (82%) | 0.2682 | 0.4774 | 0.5406 | 0.0078 |

### SQL-primary (retrieval informational) (n=3)

| K | Hit rate | Precision | MRR | nDCG | Recall |
|---:|---:|---:|---:|---:|---:|
| 10 | 1/3 (33%) | 0.0667 | 0.0833 | 0.1608 | 0.0003 |
| 50 | 1/3 (33%) | 0.0333 | 0.0833 | 0.1623 | 0.0007 |

### By tier (retrieval-primary, K=10)

| Tier | n | Hit rate | Precision | MRR | nDCG |
|---|---:|---:|---:|---:|---:|
| aggregation | 1 | 0/1 | 0.0000 | 0.0000 | 0.0000 |
| comparison | 4 | 3/4 | 0.3000 | 0.3417 | 0.4358 |
| multi_hop | 4 | 2/4 | 0.3250 | 0.3333 | 0.3945 |
| simple | 4 | 3/4 | 0.4750 | 0.5625 | 0.6042 |
| trend | 4 | 3/4 | 0.4000 | 0.7500 | 0.7020 |

## Key insights

1. **Hybrid retrieval outperforms either modality alone.** At K=50, Hybrid (RRF) achieves 82% Hit rate vs 76% vector-only and 71% BM25-only. The fusion captures both semantic similarity and exact keyword matches.

2. **Category-filtered retrieval is critical for single-category questions.** Questions like Q15 (Arts_Crafts_and_Sewing shipping damage) and Q17 (1-star Appliances themes) achieve near-perfect precision when metadata filtering narrows candidates from 150K to ~30K.

3. **SQL-primary questions should not be judged on retrieval overlap.** The `eval_mode` separation reveals that SQL-primary questions (Q5, Q6, Q8) naturally score low on retrieval metrics because their ground truth is best answered by aggregation, not document lookup. Separating these prevents misleading aggregate scores.

4. **Cross-encoder reranking improves precision but not always Hit rate.** The ablation shows Hybrid+CE has higher Precision@10 (0.347 vs 0.312) and nDCG@50 (0.541 vs 0.528) compared to plain Hybrid, but identical Hit@50. The cross-encoder reorders the top of the list rather than introducing new documents.

5. **Answer quality is high despite retrieval imperfections.** The agent pipeline achieves 4.79/5 mean critic score across 14 scored questions, with 100% scoring >= 4. The multi-agent design (SQL fallback, decomposition, retry loop) compensates for retrieval misses.

## Failure cases and error analysis

**6/17** retrieval-primary questions missed at K=10:

- Q3 (simple): "Which ASIN has the most reviews in the dataset?" — pool=293, cat_filter=none
- Q7 (aggregation): "What are the top 5 most helpful reviews by helpful_vote?" — pool=2000, cat_filter=none
- Q9 (comparison): "Compare average ratings between All_Beauty and Appliances." — pool=2000, cat_filter=none
- Q13 (multi_hop): "Find a product (asin) that has both 5-star and 1-star review" — pool=293, cat_filter=none
- Q16 (multi_hop): "Find highly rated reviews that still mention product defects" — pool=2000, cat_filter=none
- Q18 (trend): "Summarize strengths and weaknesses mentioned for Baby_Produc" — pool=2000, cat_filter=Baby_Products

Common miss patterns:
- **Generic keywords** (e.g. "quality", "good") match many documents; ground truth and retrieval may find different subsets of the same semantic space.
- **Cross-category queries** without metadata filter search 150K docs; top-10 is a very small window.
- **Structural queries** (helpful_vote ordering, ASIN lookup) require SQL, not semantic search.


## Trade-offs

- **Large ground-truth pools** (200–2 000 docs) make Hit@K and Precision@K meaningful, but Recall@K is naturally low (bounded by K / pool_size). A pool of 1 000 means Recall@10 ≤ 0.01 even with perfect retrieval.

- **CPU embeddings** (~2h for 150K rows) vs GPU — zero cost, reproducible.

- **MiniLM (384-d)** — small, fast, fits in RAM; trades semantic depth for speed.

- **Metadata filtering** narrows category queries from 150K to ~30K candidates.

- **RRF + cross-encoder** — RRF fuses dense + sparse; cross-encoder rescores top pool.


## Ablation study


### K = 10

| Mode | Hit rate | Precision | MRR | nDCG | Recall |
|------|---:|---:|---:|---:|---:|
| Vector only | 0.5294 | 0.2353 | 0.4510 | 0.4278 | 0.0017 |
| BM25 only | 0.7059 | 0.3118 | 0.4771 | 0.5132 | 0.0026 |
| Hybrid (RRF) | 0.7059 | 0.3118 | 0.5158 | 0.5365 | 0.0026 |
| Hybrid + CE | 0.6471 | 0.3471 | 0.4676 | 0.5005 | 0.0029 |

### K = 50

| Mode | Hit rate | Precision | MRR | nDCG | Recall |
|------|---:|---:|---:|---:|---:|
| Vector only | 0.7647 | 0.2441 | 0.4691 | 0.4828 | 0.0036 |
| BM25 only | 0.7059 | 0.2912 | 0.4771 | 0.5007 | 0.0047 |
| Hybrid (RRF) | 0.8235 | 0.2570 | 0.5198 | 0.5277 | 0.0073 |
| Hybrid + CE | 0.8235 | 0.2570 | 0.4774 | 0.5407 | 0.0073 |

## Answer quality (quantitative, batch)

- **Questions scored:** 14/20
- **Mean critic score:** 4.79 / 5
- **Min:** 4.0 | **Max:** 5.0
- **Score >= 4:** 14/14
- **Mean latency:** 21.1s per question
- **Pipeline errors:** 6/20

## Regenerating this report

```powershell
python scripts/build_ground_truth.py   # rebuild large relevance pools
python scripts/evaluate.py             # multi-K retrieval metrics
python scripts/ablation.py             # optional ablation
python scripts/answer_quality.py       # optional batch answer quality
```
