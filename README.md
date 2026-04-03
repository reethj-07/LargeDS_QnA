# Big Data Q&A: E-commerce Analytics Agent System

AI agent pipeline over **150K+ Amazon product reviews** (McAuley-Lab `Amazon-Reviews-2023`, loaded via **direct JSONL download** from Hugging Face). The stack combines **FAISS** (dense vectors), **BM25** (keywords), **DuckDB** (SQL analytics), and a **multi-agent** flow (**Planner → Retriever → Analyst → Critic**) using **Groq** (free tier) and **LangGraph**.

## Requirements

- Python 3.11+ (tested on 3.13)
- ~15GB disk for caches + indices (first run downloads category JSONL files)
- **Free Groq API key** — [console.groq.com](https://console.groq.com)

## Quick start

```powershell
cd C:\Projects\BigData_QnA
pip install -r requirements.txt
copy .env.example .env
# Edit .env and set GROQ_API_KEY=...
```

### 1) Ingest data and build indices

Default: **5 categories × 30,000 reviews = 150,000 rows** (may take **30–90+ minutes** on CPU for embedding).

```powershell
python scripts/ingest.py
```

Smaller smoke test:

```powershell
python scripts/ingest.py --rows-per-category 100
```

Outputs (gitignored):

- `data/processed/reviews.parquet`
- `data/indices/reviews.duckdb`
- `data/indices/faiss.index` + `data/indices/faiss_meta.json`
- `data/indices/bm25.pkl`

### 2) Ask a question (CLI)

```powershell
python scripts/query.py "What is the average rating by category?"
```

### 3) Web UI (Gradio)

```powershell
python -m src.ui.app
```

Open the printed local URL (default `http://127.0.0.1:7860`).

### 4) Retrieval evaluation

Populate `ground_truth_ids` in `evaluation/test_questions.json` (DuckDB `reviews.id` values) for meaningful Recall@K / Precision@K.

```powershell
python scripts/evaluate.py
```

Writes `evaluation/cache/eval_results.json` and refreshes `EVALUATION_REPORT.md`.

## Project layout

| Path | Role |
|------|------|
| `src/ingestion/` | Download JSONL, preprocess, orchestration |
| `src/embeddings/` | `sentence-transformers` embeddings |
| `src/storage/` | DuckDB, FAISS, BM25 persistence |
| `src/retrieval/` | Hybrid search + RRF |
| `src/agents/` | Planner, retrieval step, Analyst, Critic, LangGraph |
| `src/ui/` | Gradio app |
| `evaluation/` | Questions, metrics, batch eval |
| `scripts/` | `ingest.py`, `query.py`, `evaluate.py` |

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — diagrams and data flow
- [EVALUATION_REPORT.md](EVALUATION_REPORT.md) — metrics (after you run eval)
- [ANY_OTHER_DIAGRAMS.md](ANY_OTHER_DIAGRAMS.md) — extra figures / exports

## Demo video

Record a short walkthrough (complex query + a failure case) and link it in `DEMO_VIDEO/README.md` or your submission.

## Assignment checklist (from brief)

- Data platform: ingestion + **vector (FAISS)** + **hybrid (SQL + vector + BM25)**
- Retrieval: embeddings + similarity; bonus hybrid search implemented
- Agents: **Planner**, **Retriever**, **Analyst**, **Critic** (LangGraph)
- Evaluation: **Recall@K**, **Precision@K** (when ground truth filled) + qualitative notes
- Observability: structured logs under `logs/app.log`

## License

Educational / assignment use. Dataset subject to McAuley-Lab / Amazon license on Hugging Face.
