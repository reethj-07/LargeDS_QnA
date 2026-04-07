# Big Data Q&A: E-commerce Analytics Agent System

AI **multi-agent** pipeline over **150K+ Amazon product reviews** (McAuley-Lab `Amazon-Reviews-2023`, loaded via **direct JSONL download** from Hugging Face). The stack combines **FAISS** (dense vectors), **BM25** (keywords), **DuckDB** (SQL analytics), **cross-encoder reranking**, and a **multi-agent** flow with **dynamic routing** (**Planner -> Router -> [Decompose | SQL-first | Direct] -> Analyst -> Critic**) using **LangGraph**. LLM calls use **Google Gemini** (AI Studio) when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set, with **Groq** as automatic **fallback** on errors or rate limits; with no Gemini key, the app uses **Groq only**.

## Dataset description (assignment deliverable)

| Item | Detail |
|------|--------|
| **Source** | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) on Hugging Face Hub |
| **Access** | Raw JSONL files under `raw/review_categories/*.jsonl` via `huggingface_hub.hf_hub_download` (the `datasets` Python API does not load this repo due to deprecated loading scripts). |
| **Subset used** | 5 categories x 30,000 reviews = **150,000 rows**: All_Beauty, Appliances, Amazon_Fashion, Arts_Crafts_and_Sewing, Baby_Products |
| **Fields stored** | `id`, `asin`, `category`, `rating`, `title`, `text`, `doc_text`, `timestamp`, `helpful_vote`, `verified_purchase` |
| **Preprocessing** | Normalize Unicode (NFKC), strip HTML-like tags, build `doc_text = title + " " + text`, coerce `verified_purchase` to bool, drop empty rows |
| **Artifacts** | Parquet (`data/processed/`), DuckDB + FAISS + BM25 (`data/indices/`) -- large files gitignored; rebuild with `python scripts/ingest.py` |

## Requirements

- Python 3.11+ (tested on 3.13)
- ~15GB disk for caches + indices (first run downloads category JSONL files)
- At least one LLM key:
  - **Gemini (recommended primary):** [Google AI Studio](https://aistudio.google.com/apikey) — set `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - **Groq (recommended fallback):** [console.groq.com](https://console.groq.com) — set `GROQ_API_KEY`

With **both** keys, Gemini is invoked first; if the call fails (quota, 429, network, invalid response path, etc.), LangChain automatically retries with Groq. Gemini’s free tier can be tight on **RPM/RPD** for multi-agent flows, so keeping Groq configured avoids hard failures.

Defaults use **`gemini-2.5-flash`** / **`gemini-2.5-pro`** (older `gemini-1.5-*` IDs often return **404** on the current API). Override with `MODEL_GEMINI_PLANNER`, `MODEL_GEMINI_ANALYST`, `MODEL_GEMINI_CRITIC` if your project lists different model strings in [Google’s model docs](https://ai.google.dev/gemini-api/docs/models/gemini).

## Quick start

```powershell
cd C:\Projects\BigData_QnA
pip install -r requirements.txt
copy .env.example .env
# Edit .env: set GEMINI_API_KEY or GOOGLE_API_KEY, and GROQ_API_KEY for fallback (or Groq-only).
# Optional: LLM_PRIMARY=groq to force Groq and skip Gemini.
```

### 1) Ingest data and build indices

Default: **5 categories x 30,000 reviews = 150,000 rows** (may take **30-90+ minutes** on CPU for embedding).

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

Open the printed local URL (default `http://localhost:7860`).

### 4) Retrieval evaluation

After ingestion, rebuild ground truth from DuckDB and run the evaluation pipeline:

```powershell
python scripts/build_ground_truth.py   # deterministic, content-aligned ground truth
python scripts/evaluate.py             # Recall, Precision, MRR, nDCG, Hit@K
```

`scripts/evaluate.py` runs the retrieval batch with **eval_mode** awareness (retrieval-primary vs SQL-primary questions) and **category-filtered retrieval** for single-category queries, then regenerates `EVALUATION_REPORT.md`.

### 5) Ablation study

Compare retrieval modes side-by-side (vector-only, BM25-only, hybrid, hybrid + cross-encoder):

```powershell
python scripts/ablation.py
```

### 6) Batch answer quality

Run all 20 test questions through the full agent pipeline, collect critic scores and latency:

```powershell
python scripts/answer_quality.py
```

### 6b) SQL accuracy (aggregation questions)

Generate reference answers for SQL-primary questions:

```powershell
python scripts/sql_eval.py
```

### 7) Unit tests

**30+ tests** covering metrics, RRF, preprocessor, SQL safety, decomposer, and synthesizer:

```powershell
python -m pytest -v
```

CI (`.github/workflows/ci.yml`) runs tests with `USE_CROSS_ENCODER=0`.

### 8) Submission

- [ ] Repo **public** *or* **private** with collaborator `uptiq-chaitanya` (read) per assignment.
- [ ] Submit **Google Form** with repo link: [forms.gle/9iPeUBHKcdHhuSq67](https://forms.gle/9iPeUBHKcdHhuSq67).
- [ ] Record **demo video** (complex query + failure case) when ready -- link in **`DEMO_VIDEO/README.md`**.

## Multi-agent architecture

The system uses **6 specialized agents** orchestrated by LangGraph with dynamic routing:

```
User Query
    |
    v
[Planner] -- classifies query type + decides route
    |
    v
[Router] -- conditional edges based on route
    |
    +-- "direct"    --> [Retriever] --+
    +-- "sql_first" --> [SQL-first]  --+--> [Analyst] --> [Critic] --> Answer
    +-- "decompose" --> [Decomposer] --> [Sub-Retriever] --> [Synthesizer] --+
```

| Agent | Model | Role |
|-------|-------|------|
| **Planner** | llama-3.1-8b-instant | Classify query, decide route, generate SQL hint |
| **Decomposer** | llama-3.1-8b-instant | Break multi-hop / comparison queries into sub-questions |
| **Retriever** | -- | Hybrid search (FAISS + BM25 + RRF + cross-encoder) |
| **Synthesizer** | -- | Merge per-sub-question retrieval results |
| **Analyst** | llama-3.3-70b-versatile | Generate grounded answer with citations |
| **Critic** | llama-3.3-70b-versatile | Score answer quality, trigger retry if needed |

## Deployment

### Docker

```bash
docker build -t bigdata-qna .
docker run -p 7860:7860 --env-file .env bigdata-qna
```

### Render

1. Push to GitHub.
2. Create a new **Web Service** on [Render](https://render.com), point to the repo.
3. Render auto-detects `render.yaml` (Docker runtime, persistent disk for data).
4. Set `GROQ_API_KEY` in Render environment.
5. Run ingestion once via Render shell: `python scripts/ingest.py --rows-per-category 100`.

### Hugging Face Spaces (alternative)

Gradio apps deploy directly to HF Spaces -- push the repo and set secrets.

## Observability (assignment bonus)

### Structured JSON logs

Structured **JSON logs** (stdout + `logs/app.log`) via `src/observability/logger.py` with **trace_id** correlation:

| Event | What is logged |
|-------|----------------|
| `pipeline_query` | User question length + preview + trace_id |
| `agent_planner` | `query_type`, `route` |
| `agent_decomposer` | Sub-question count + text |
| `retrieval_sql` / `retrieval_sql_error` | SQL row count or error |
| `retrieval_hybrid` | Doc count, retrieved `reviews.id` list (up to 30), fusion scores |
| `agent_analyst` | Answer length |
| `agent_critic` | Score, `needs_retry` |
| `ui_query` | Query length (Gradio) |

### LangSmith tracing (optional)

Uncomment in `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your_key_here
LANGCHAIN_PROJECT=bigdata-qna
```

All LangChain/LangGraph calls are **automatically traced** in LangSmith (free tier).

### OpenTelemetry tracing (optional)

```env
OTEL_ENABLED=true
OTEL_EXPORTER=console   # or "otlp" for a collector
OTEL_ENDPOINT=http://localhost:4317
```

Per-agent spans with attributes (query, doc_count, latency) exported to console or OTLP.

## Project layout

| Path | Role |
|------|------|
| `src/ingestion/` | Download JSONL, preprocess, orchestration |
| `src/embeddings/` | `sentence-transformers` embeddings |
| `src/storage/` | DuckDB, FAISS, BM25 persistence |
| `src/retrieval/` | Hybrid search + RRF + optional cross-encoder rerank |
| `src/agents/` | Planner, Decomposer, Retriever, Synthesizer, Analyst, Critic, LangGraph |
| `src/llm/` | Gemini + Groq routing (`get_chat_llm`, fallbacks) |
| `src/observability/` | Structured logging + OTel tracing |
| `src/ui/` | Gradio app with tabbed trace/observability UI |
| `evaluation/` | Questions, metrics, batch eval, ablation, answer quality, SQL accuracy |
| `scripts/` | `ingest.py`, `query.py`, `evaluate.py`, `ablation.py`, `answer_quality.py`, `sql_eval.py` |
| `tests/` | Unit tests (metrics, RRF, preprocessor, SQL safety, decomposer, synthesizer) |
| `.github/workflows/` | CI: pytest on every push / PR |
| `Dockerfile` | Container build for deployment |
| `render.yaml` | Render deployment config |

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) -- diagrams and data flow
- [ADVANCED.md](ADVANCED.md) -- retrieval theory (RRF, cross-encoder), evaluation nuance, talking points
- [EVALUATION_REPORT.md](EVALUATION_REPORT.md) -- metrics (after you run eval)
- [ANY_OTHER_DIAGRAMS.md](ANY_OTHER_DIAGRAMS.md) -- sequence diagrams, retrieval fusion, evaluation pipeline

## Demo video

Record a short walkthrough (complex query + a failure case) and paste the link in [`DEMO_VIDEO/README.md`](DEMO_VIDEO/README.md). **Optional until you record it** -- everything else can be submitted without the video.

## Assignment checklist (from brief)

- Data platform: ingestion + **vector (FAISS)** + **hybrid (SQL + vector + BM25)**
- Retrieval: embeddings + similarity; bonus hybrid search + reranking + **metadata filtering**
- Agents: **Planner**, **Decomposer**, **Retriever**, **Synthesizer**, **Analyst**, **Critic** (LangGraph multi-agent)
- Query complexity: multi-hop (decomposition), aggregations (SQL-first), comparisons (decompose per entity)
- Evaluation:
  - **Retrieval:** Recall@K, Precision@K, Hit@K, MRR@K, nDCG@K with **eval_mode** separation
  - **Ablation study:** vector-only vs BM25 vs hybrid vs hybrid+CE
  - **Answer quality:** batch critic scores (mean 4.79/5)
  - **SQL accuracy:** reference answers for aggregation questions
  - **Category-filtered retrieval** for fair single-category evaluation
- Observability: structured logs with trace_id + optional LangSmith + optional OpenTelemetry
- Deployment: Docker + Render

## License

Educational / assignment use. Dataset subject to McAuley-Lab / Amazon license on Hugging Face.
