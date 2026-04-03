# Big Data Q&A: E-commerce Analytics Agent System

AI agent pipeline over **150K+ Amazon product reviews** (McAuley-Lab `Amazon-Reviews-2023`, loaded via **direct JSONL download** from Hugging Face). The stack combines **FAISS** (dense vectors), **BM25** (keywords), **DuckDB** (SQL analytics), and a **multi-agent** flow (**Planner → Retriever → Analyst → Critic**) using **Groq** (free tier) and **LangGraph**.

## Dataset description (assignment deliverable)

| Item | Detail |
|------|--------|
| **Source** | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) on Hugging Face Hub |
| **Access** | Raw JSONL files under `raw/review_categories/*.jsonl` via `huggingface_hub.hf_hub_download` (the `datasets` Python API does not load this repo due to deprecated loading scripts). |
| **Subset used** | 5 categories × 30,000 reviews = **150,000 rows**: All_Beauty, Appliances, Amazon_Fashion, Arts_Crafts_and_Sewing, Baby_Products |
| **Fields stored** | `id`, `asin`, `category`, `rating`, `title`, `text`, `doc_text`, `timestamp`, `helpful_vote`, `verified_purchase` |
| **Preprocessing** | Normalize Unicode (NFKC), strip HTML-like tags, build `doc_text = title + " " + text`, coerce `verified_purchase` to bool, drop empty rows |
| **Artifacts** | Parquet (`data/processed/`), DuckDB + FAISS + BM25 (`data/indices/`) — large files gitignored; rebuild with `python scripts/ingest.py` |

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

After ingestion, optionally generate labels and short `retrieval_query` strings (see `evaluation/test_questions.json`):

```powershell
python scripts/build_ground_truth.py
python scripts/evaluate.py
```

`scripts/evaluate.py` runs the retrieval batch and regenerates `EVALUATION_REPORT.md`

(`evaluation/cache/` is gitignored; the report in the repo documents methodology and metrics).

### 5) Submission (no video yet)

- [ ] Repo **public** *or* **private** with collaborator `uptiq-chaitanya` (read) per assignment.
- [ ] Submit **Google Form** with repo link: [forms.gle/9iPeUBHKcdHhuSq67](https://forms.gle/9iPeUBHKcdHhuSq67).
- [ ] Record **demo video** (complex query + failure case) when ready — link in **`DEMO_VIDEO/README.md`**.

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

Record a short walkthrough (complex query + a failure case) and paste the link in [`DEMO_VIDEO/README.md`](DEMO_VIDEO/README.md). **Optional until you record it** — everything else can be submitted without the video.

## Observability (assignment bonus)

Structured **JSON logs** (stdout + `logs/app.log`) via `src/observability/logger.py`:

| Event | What is logged |
|-------|----------------|
| `pipeline_query` | User question length + preview (first 400 chars) |
| `ingestion_*` | Load progress, parquet/DuckDB/BM25/FAISS paths and row counts |
| `agent_planner` | `query_type` from Planner |
| `retrieval_sql` / `retrieval_sql_error` | SQL row count or error message |
| `retrieval_hybrid` | Doc count, **retrieved `reviews.id` list** (up to 30), sample fusion scores |
| `agent_analyst` | Answer length |
| `agent_critic` | Score, `needs_retry` |
| `ui_query` | Query length (Gradio) |

**Not included (optional):** LangSmith / OpenTelemetry tracing — would need API keys and extra deps; file + JSON logs cover the brief’s logging requirements.

## Assignment checklist (from brief)

- Data platform: ingestion + **vector (FAISS)** + **hybrid (SQL + vector + BM25)**
- Retrieval: embeddings + similarity; bonus hybrid search implemented
- Agents: **Planner**, **Retriever**, **Analyst**, **Critic** (LangGraph)
- Evaluation: **Recall@K**, **Precision@K** (when ground truth filled) + qualitative notes
- Observability: structured logs under `logs/app.log` (see **Observability** above)

## License

Educational / assignment use. Dataset subject to McAuley-Lab / Amazon license on Hugging Face.
