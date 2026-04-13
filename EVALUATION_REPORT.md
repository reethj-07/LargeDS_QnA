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
| 1 | simple | both | 2000 | 1 | 0.4000 | 0.3333 | 0.5937 | 0.0020 | All_Beauty |
| 2 | simple | retrieval | 495 | 1 | 0.5000 | 0.5000 | 0.6810 | 0.0101 |  |
| 3 | simple | both | 293 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 4 | simple | retrieval | 2000 | 1 | 0.8000 | 1.0000 | 0.9794 | 0.0040 |  |
| 5 | aggregation | sql | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 6 | aggregation | sql | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 7 | aggregation | both | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 8 | aggregation | sql | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 9 | comparison | both | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 10 | comparison | both | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 11 | comparison | both | 2000 | 1 | 0.1000 | 0.1429 | 0.3333 | 0.0005 |  |
| 12 | comparison | both | 1925 | 1 | 0.9000 | 1.0000 | 0.9972 | 0.0047 |  |
| 13 | multi_hop | both | 293 | 1 | 0.1000 | 0.1250 | 0.3155 | 0.0034 |  |
| 14 | multi_hop | retrieval | 2000 | 1 | 0.6000 | 1.0000 | 0.9538 | 0.0030 |  |
| 15 | multi_hop | retrieval | 1220 | 1 | 0.8000 | 0.5000 | 0.8165 | 0.0066 | Arts_Crafts_and_Sewing |
| 16 | multi_hop | retrieval | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |
| 17 | trend | retrieval | 963 | 1 | 0.9000 | 1.0000 | 0.9972 | 0.0093 | Appliances |
| 18 | trend | retrieval | 2000 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Baby_Products |
| 19 | trend | retrieval | 1251 | 1 | 0.4000 | 1.0000 | 0.9558 | 0.0032 |  |
| 20 | trend | retrieval | 2000 | 1 | 0.3000 | 1.0000 | 0.7530 | 0.0015 |  |

## Aggregate metrics


### All questions (n=20)

| K | Hit rate | Precision | MRR | nDCG | Recall |
|---:|---:|---:|---:|---:|---:|
| 10 | 11/20 (55%) | 0.2900 | 0.3801 | 0.4188 | 0.0024 |
| 50 | 15/20 (75%) | 0.2255 | 0.3910 | 0.4598 | 0.0070 |

### Retrieval-primary (eval_mode != sql) (n=17)

| K | Hit rate | Precision | MRR | nDCG | Recall |
|---:|---:|---:|---:|---:|---:|
| 10 | 11/17 (65%) | 0.3412 | 0.4471 | 0.4927 | 0.0028 |
| 50 | 14/17 (82%) | 0.2594 | 0.4561 | 0.5208 | 0.0081 |

### SQL-primary (retrieval informational) (n=3)

| K | Hit rate | Precision | MRR | nDCG | Recall |
|---:|---:|---:|---:|---:|---:|
| 10 | 0/3 (0%) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 50 | 1/3 (33%) | 0.0333 | 0.0222 | 0.1138 | 0.0007 |

### By tier (retrieval-primary, K=10)

| Tier | n | Hit rate | Precision | MRR | nDCG |
|---|---:|---:|---:|---:|---:|
| aggregation | 1 | 0/1 | 0.0000 | 0.0000 | 0.0000 |
| comparison | 4 | 2/4 | 0.2500 | 0.2857 | 0.3326 |
| multi_hop | 4 | 3/4 | 0.3750 | 0.4062 | 0.5215 |
| simple | 4 | 3/4 | 0.4250 | 0.4583 | 0.5635 |
| trend | 4 | 3/4 | 0.4000 | 0.7500 | 0.6765 |

## Failure cases and error analysis

**6/17** retrieval-primary questions missed at K=10:

- Q3 (simple): "Which ASIN has the most reviews in the dataset?" — pool=293, cat_filter=none
- Q7 (aggregation): "What are the top 5 most helpful reviews by helpful_vote?" — pool=2000, cat_filter=none
- Q9 (comparison): "Compare average ratings between All_Beauty and Appliances." — pool=2000, cat_filter=none
- Q10 (comparison): "Compare customer sentiment in Amazon_Fashion vs Baby_Product" — pool=2000, cat_filter=none
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

- **Questions scored:** 20/20
- **Mean critic score:** 4.20 / 5
- **Min:** 2.0 | **Max:** 5.0
- **Score >= 4:** 18/20
- **Mean latency:** 55.8s per question
- **Pipeline errors:** 0/20 (none)

## Regenerating this report

```powershell
python scripts/build_ground_truth.py   # rebuild large relevance pools
python scripts/evaluate.py             # multi-K retrieval metrics
python scripts/ablation.py             # optional ablation
python scripts/answer_quality.py       # optional batch answer quality
```
