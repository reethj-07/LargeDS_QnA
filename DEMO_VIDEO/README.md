# Demo Video

**Where to put your link:** Edit this file and replace the placeholder on the line **`Demo URL:`** in the [Demo URL](#demo-url) section below (same place the assignment README points to). Use a Loom, YouTube (unlisted), Google Drive share link, or any URL graders can open.

**Status:** Record when ready.

Record a **Loom** or local MP4 showing the following three scenarios:

## Required scenes

### 1. Data ingestion (brief)
- Show `python scripts/ingest.py` running (or mention that indices are prebuilt)
- Mention: 5 categories × 30K = 150K Amazon reviews

### 2. Complex query (main demo)
- Launch UI: `python -m src.ui.app`
- Ask a **multi-hop or comparison query**, e.g.:
  - "Compare average ratings across categories and name one theme from reviews in the lowest-rated category."
  - "Find highly rated reviews that still mention product defects."
- Show all three tabs: **Answer**, **Agent Trace** (routing, sub-questions, critic score), **Retrieved Documents**
- Highlight: Planner chose route → Decomposer broke into sub-questions → SQL + retrieval → Analyst cited `[id=...]` → Critic scored ≥ 4

### 3. Failure case
- Ask a query that triggers low critic score or empty SQL, e.g.:
  - "What is the revenue by product?" (revenue column doesn't exist)
  - "List all reviews from a discontinued brand." (vague, no clear category)
- Show how the system gracefully handles: SQL skipped, falls back to retrieval, critic flags low grounding

### 4. Evaluation (optional)
- Show `python scripts/evaluate.py` running and the output metrics
- Open `EVALUATION_REPORT.md` and explain the eval_mode split

## Recording tips

- Keep it under 5 minutes
- Use a clear screen resolution (1080p+)
- Narrate what you're doing and why
- Paste the link below when done

## Demo URL

Paste your recording here (this is the link reviewers open):

- **Demo URL:** _add your link here_
