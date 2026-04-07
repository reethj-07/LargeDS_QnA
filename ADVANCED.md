# Advanced topics (submission depth)

This document summarizes design decisions, retrieval theory, and how to demonstrate expertise in reviews or interviews.

## End-to-end data path

1. **Streamed JSONL** from Hugging Face Hub (`hf_hub_download`) into category files under `data/raw/`.
2. **Normalize and denoise** text (Unicode NFKC, strip markup) and build `doc_text` for embedding and BM25.
3. **Columnar storage:** Parquet for bulk; **DuckDB** for SQL and row fetch by `id` at answer time.
4. **Dense index:** `sentence-transformers/all-MiniLM-L6-v2` -> FAISS `IndexFlatIP` (cosine via normalized vectors + inner product).
5. **Sparse index:** BM25 over the same tokenized `doc_text` corpus.
6. **Fusion:** **Reciprocal Rank Fusion (RRF)** merges vector and keyword rankings without calibrating scores across modalities.
7. **Rerank (CPU):** Top `RERANK_POOL` RRF candidates are scored by a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which jointly encodes query+document and is stronger than dot-product similarity for **precision at the top** of the list.

## Why RRF then cross-encoder?

- **Bi-encoder (MiniLM)** is fast and scales to millions of vectors; it is a *rough* filter.
- **RRF** is cheap and robust when dense and sparse lists disagree (e.g. exact product name vs. paraphrase).
- **Cross-encoder** is slow per pair but only runs on tens of candidates, reordering the shortlist for the LLM context window.

Disable reranking with `USE_CROSS_ENCODER=0` for ablations or CI.

## Multi-agent orchestration

- **Planner:** Intent + optional SQL; constraints from schema.
- **Retriever:** Hybrid search + citations; logs structured events for observability.
- **Analyst:** Grounded answer from SQL rows + retrieved snippets (Groq).
- **Critic:** Scores grounding; loop or stop based on threshold.

LangGraph keeps state explicit (`AgentState`) and makes retries and tracing auditable.

## Evaluation nuance

- **Ground truth** uses deterministic, content-aligned SQL (keyword `LIKE` filters matching what retrieval would find). No `ORDER BY random()` — results are reproducible across runs.
- **Eval modes** separate questions by answer strategy:
  - `retrieval` — metrics measure document overlap directly
  - `sql` — answer correctness comes from SQL; retrieval overlap is informational
  - `both` — SQL + retrieval together
- **Category-filtered retrieval:** Single-category questions pre-filter candidates via DuckDB metadata, narrowing the search space from 150K to ~30K and boosting precision.
- **Category in doc_text:** Preprocessing prefixes each review with `[Category Name]` so BM25 and FAISS can leverage category signal during retrieval.
- **Hit@K**, **MRR@K**, and **nDCG@K** complement recall: they reward *any* good hit and *rank* of the first hit, which aligns better with user experience when K is small.
- Report metrics **before/after** changes (e.g. rerank on/off, `retrieval_query` tuning), not as absolute ceilings.

## Ablation study

`python scripts/ablation.py` compares four retrieval modes on the same 20 test questions:

| Mode | What it tests |
|------|---------------|
| Vector only | Dense (FAISS) without keyword backup |
| BM25 only | Sparse without semantic understanding |
| Hybrid (RRF) | Fusion of both ranked lists |
| Hybrid + CE | RRF pool rescored by cross-encoder |

Expected pattern: Hybrid >= either alone; CE may help precision at the top of the list (MRR, nDCG) more than bulk recall, especially for short keyword queries.

## Batch answer quality

`python scripts/answer_quality.py` runs all 20 questions through the full `run_agent_pipeline` (Planner -> Retriever -> Analyst -> Critic) and records:

- **Critic score** (1-5): quantitative answer grounding metric
- **Latency**: end-to-end time per question
- **Retries**: how often the critic triggered a retrieval retry
- **Errors**: any pipeline failures

This closes the "Answer quality" quantitative requirement from the brief.

## Test suite

**30+ unit tests** in `tests/` covering:

| File | What it tests |
|------|---------------|
| `test_metrics.py` | Recall@K, MRR@K, nDCG@K |
| `test_metrics_extended.py` | AP, Hit@K, edge cases |
| `test_rrf.py` | Reciprocal rank fusion ordering |
| `test_preprocessor.py` | Text cleaning, boolean/float/int parsing, row filtering, category prefix |
| `test_sql_safety.py` | SQL injection prevention (INSERT, DROP, multiple statements) |
| `test_decomposer.py` | JSON extraction and decomposer mocks |
| `test_planner_route.py` | Planner routing mocks |
| `test_synthesizer.py` | Sub-result merging, dedup, ordering |
| `test_tracing.py` | Trace IDs and noop spans |
| `test_llm_chat.py` | API key checks and LLM routing |

## Operations

- **Observability:** JSON logs (`src/observability/logger.py`) for pipeline steps and retrieval summaries.
- **CI:** GitHub Actions runs `pytest` with `USE_CROSS_ENCODER=0` so tests do not download reranker weights.

## Suggested "show off" talking points

1. Trade-off: **150K rows on CPU** vs cloud GPU -- documented ingest time and index sizes.
2. **Hybrid retrieval** is not redundant: vector and BM25 capture different failure modes -- the ablation quantifies this.
3. **SQL + RAG:** aggregates from DuckDB vs qualitative evidence from reviews -- when each is authoritative.
4. **Evaluation honesty:** eval_mode separation acknowledges that SQL-primary questions shouldn't be judged solely on retrieval overlap.
5. **Content-aligned ground truth** with deterministic SQL ensures reproducible, meaningful metrics.
6. **Metadata-filtered retrieval** for category-specific queries — mirrors production RAG systems.
7. **Cross-encoder rerank** boosts precision without GPU -- show the ablation table.
8. **30+ tests + CI** -- demonstrates production mindset, not just a notebook prototype.
9. **Batch answer quality** -- quantitative LLM-as-judge (critic scores), not just ad-hoc spot checks.
10. **SQL accuracy evaluation** -- separate eval track for aggregation questions.
