# Evaluation Report

## Retrieval (hybrid FAISS + BM25 + RRF)

| id | tier | Recall@K | Precision@K | has_ground_truth |
|---:|---|---:|---:|:---:|
| 1 | simple | N/A | N/A | False |
| 2 | simple | N/A | N/A | False |
| 3 | simple | N/A | N/A | False |
| 4 | simple | N/A | N/A | False |
| 5 | aggregation | N/A | N/A | False |
| 6 | aggregation | N/A | N/A | False |
| 7 | aggregation | N/A | N/A | False |
| 8 | aggregation | N/A | N/A | False |
| 9 | comparison | N/A | N/A | False |
| 10 | comparison | N/A | N/A | False |
| 11 | comparison | N/A | N/A | False |
| 12 | comparison | N/A | N/A | False |
| 13 | multi_hop | N/A | N/A | False |
| 14 | multi_hop | N/A | N/A | False |
| 15 | multi_hop | N/A | N/A | False |
| 16 | multi_hop | N/A | N/A | False |
| 17 | trend | N/A | N/A | False |
| 18 | trend | N/A | N/A | False |
| 19 | trend | N/A | N/A | False |
| 20 | trend | N/A | N/A | False |

## Notes

- Populate `ground_truth_ids` in `evaluation/test_questions.json` with relevant `reviews.id` values (from DuckDB) to enable Recall@K / Precision@K.
- Answer quality: use Groq LLM-as-judge in the demo or add a separate script; not included in default batch to save rate limits.
