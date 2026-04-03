# System Architecture

## End-to-end flow

```mermaid
flowchart TD
    subgraph ingest [Ingestion one-time]
        HF[HF Hub JSONL] --> Load[Loader per category]
        Load --> Prep[Preprocessor]
        Prep --> DDB[DuckDB reviews]
        Prep --> EMB[MiniLM embeddings]
        EMB --> FAISS[FAISS IndexFlatIP]
        Prep --> BM25[BM25Okapi]
    end

    subgraph online [Query online]
        Q[User query] --> P[Planner Groq]
        P --> R[Retriever hybrid]
        R --> V[FAISS search]
        R --> B[BM25 search]
        R --> S[Optional DuckDB SQL]
        V --> RRF[RRF merge]
        B --> RRF
        RRF --> A[Analyst Groq]
        A --> C[Critic Groq]
        C -->|retry if low score| R
        C --> O[Answer + trace]
    end
```

## Components

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| Data | Hugging Face `huggingface_hub` | Download `raw/review_categories/*.jsonl` |
| Structured store | DuckDB | Filters, aggregations, joins (`reviews` table) |
| Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2` + FAISS | Cosine similarity via normalized inner product |
| Sparse retrieval | `rank_bm25` | Keyword / lexical match |
| Fusion | Reciprocal Rank Fusion (RRF) | Merge ranked lists without score calibration |
| Planner | `llama-3.1-8b-instant` (Groq) | Query type + plan + optional `SELECT` |
| Analyst | `llama-3.3-70b-versatile` (Groq) | Grounded answer with `[id=]` citations |
| Critic | `llama-3.3-70b-versatile` (Groq) | JSON score + retry signal |
| Orchestration | LangGraph `StateGraph` | Planner → retrieve → analyst → critic → optional retry |

## Observability

- JSON lines to stdout and `logs/app.log` via `src/observability/logger.py`
- Gradio shows agent trace + retrieved ids

## Security

- `SqlStore.query_safe` allows only single-statement `SELECT` on `reviews` (no DDL/DML).
