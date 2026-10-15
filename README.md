# Education Video AI Assistant

An AI-powered retrieval-augmented assistant for lecture videos. It downloads or accepts a video, extracts audio, transcribes speech with Whisper, extracts key frames, runs OCR, stores transcript and frame evidence in Chroma, and answers student questions using a LangChain retrieval and LLM pipeline backed by a FastAPI REST API.

---

## Folder Structure

```text
Education-Video-AI-Assistant/
|-- app.py                      # CLI entry point
|-- main.py                     # FastAPI app factory / ASGI entry point
|-- config.py                   # Single source of truth for all config knobs
|-- requirements.txt
|-- README.md
|-- Dockerfile
|-- .dockerignore
|-- .env                        # Local secrets (never committed)
|-- api/
|   |-- routes/
|   |   |-- auth_router.py      # /api/auth — register, login
|   |   |-- video_router.py     # /api/videos — upload, ingest, status
|   |   `-- chat_router.py      # /api/chat — sessions, messages, Q&A
|   |-- auth_dependencies.py    # FastAPI dependency: resolve principal
|   |-- auth_schema.py          # AuthPrincipal, TokenResponse, …
|   |-- schemas.py              # Request/response Pydantic models
|   `-- exceptions.py           # HTTP exception helpers
|-- services/
|   |-- auth_service.py         # Register, login, JWT, password hashing
|   |-- chat_service.py         # ask_question() orchestration
|   |-- session_service.py      # Session CRUD with ownership enforcement
|   |-- retrieval_service.py    # Load retrieval components per video
|   `-- video_services.py       # Video ingest orchestration
|-- database/
|   |-- connection.py           # psycopg3 connection pool
|   `-- queries/
|       |-- user_query.py
|       |-- chat_session_query.py
|       |-- chat_message_query.py   # Ownership-enforced message writes
|       |-- memory_query.py         # Ownership-enforced message reads
|       `-- videos_query.py
|-- src/
|   |-- ingest.py
|   |-- retrieval.py
|   |-- chat.py                 # lcel_chat() pipeline (Option 2) + chat()
|   |-- memory.py               # Sliding-window memory + LLM summary
|   |-- llm.py
|   |-- prompts.py
|   |-- base_models.py          # Answer, Summary, Decision, …
|   |-- ocr.py
|   |-- speech.py
|   |-- frame_extractor.py
|   |-- loader.py
|   |-- logger.py
|   |-- cache.py                # Redis-backed response cache
|   `-- utils.py
|-- evaluation/
|   |-- eval_pipeline.py
|   |-- generate_qa.py
|   |-- retrieval_eval.py
|   |-- reranker_eval.py
|   |-- judge_eval.py
|   |-- result_aggregation.py
|   |-- base_model.py
|   |-- prompts.py
|   `-- utils.py
|-- evaluation_results/
|   |-- qa_pairs/
|   |-- retrieval_eval_results/
|   |-- rerank_eval_results/
|   |-- judge_eval_results/
|   |-- retrieval_eval_summary.json
|   |-- rerank_eval_summary.json
|   `-- per_video_analysis.json
|-- ui/
|   |-- streamlit_app.py
|   `-- api_client.py
|-- notebooks/
|   `-- VideoAssistant.ipynb    # Original prototype notebook (unchanged)
`-- outputs/
    `-- logs/
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration

All paths, model names, and tuning knobs (chunk sizes, retriever k values, ensemble weights, thresholds, cache TTLs) live in `config.py` — it is the single source of truth imported by every module.

API keys and secrets are loaded from a `.env` file at the repo root:

```text
# LLM providers
GEMINI_API_KEY=...            # Main answer LLM (Gemini)
GROQ_API_KEY=...              # Decision + eval LLMs (Groq)
OPENROUTER_API_KEY=...        # Conversation summary LLM
NANAROUTER_API_KEY=...        # Judge LLM (OpenAI-compatible endpoint)
NANAROUTER_BASE_URL=...

# Tracing (optional)
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=RAG_VideoAssistant

# Auth
JWT_SECRET_KEY=replace-with-a-long-random-secret
JWT_EXPIRE_MINUTES=60

# Database
DATABASE_URL=postgresql://...

# Redis cache
REDIS_URL=redis://localhost:6379

# CORS
CORS_ORIGINS=http://localhost:3000
```

---

## Authentication & Session Ownership

Authentication is optional for chat and session endpoints. Two principal types are supported:

| Principal | How to obtain | How sessions are scoped |
|---|---|---|
| **Authenticated user** | `POST /api/auth/register` then `POST /api/auth/login` → JWT bearer token | Sessions are tied to `user_id`; only that user can read or write them |
| **Anonymous user** | First request without a bearer token → server sets an HttpOnly `anonymous_owner_token` cookie | Sessions are tied to a hashed token stored in the cookie |

Ownership is enforced at **every layer**:
- The service layer verifies ownership via `get_chat_session(session_id, principal)` before any operation.
- The database query layer independently re-validates ownership using a `JOIN` + `owner_filter` clause on every read and write, so direct calls to the DB functions are also safe.
- Message writes (`create_chat_message`) reject inserts for sessions that don't belong to the caller.

Invalid or expired bearer tokens return `401 Unauthorized`.

> **Schema note:** The ownership columns (`user_id`, `anonymous_owner_token_hash`) require a one-time schema migration. In development, reset the PostgreSQL schema before first startup. Use a proper migration tool (e.g. Alembic) in production.

---

## How To Run

### CLI (notebook / local workflow)

```bash
pip install -r requirements.txt

python app.py --preprocess                            # Ingest video → transcripts, frames, Chroma DBs
python app.py --chat                                  # Interactive chat loop
python app.py --query "your question here"            # Single question
python app.py --eval                                  # Run the evaluation pipeline
python app.py --url <youtube-url> --preprocess        # Ingest a specific YouTube URL
python app.py --url <youtube-url> --chat              # Ingest then chat

python -m evaluation.result_aggregation               # Aggregate per-video eval results into summaries
```

Before running `--chat`, `--query`, or `--eval` without `--preprocess`, set `config.VIDEO_ID` and `config.VIDEO_PATH` in `config.py`.

### API Server

```bash
uvicorn main:app --reload
```

The API is available at <http://localhost:8000>. Interactive docs are at <http://localhost:8000/docs>.

---

## Docker

The recommended setup runs the FastAPI app together with PostgreSQL and Redis via Docker Compose:

```bash
cp .env.example .env
# Fill in API keys and secrets in .env
docker compose up --build
```

Health check:

```bash
curl http://localhost:8000/health
```

Video artifacts and database data are persisted in Compose volumes. Stop with `docker compose down`; remove all data with `docker compose down -v`.

The image can also run the CLI workflow:

```bash
docker build -t video-assistant .
docker run --rm -it --env-file .env -v video_data:/app/data video-assistant --chat
docker run --rm -it --env-file .env -v video_data:/app/data video-assistant --url <youtube-url> --preprocess
```

> The container defaults to CPU execution. GPU acceleration requires a separate NVIDIA runtime configuration.

---

## Preprocessing Pipeline

```text
Video
  -> Audio extraction
  -> Audio chunking
  -> Whisper transcription
  -> Transcript document chunking
  -> Text embeddings (Chroma text store, keyed by video_id)
  -> Changed-frame extraction
  -> CLIP frame embeddings (Chroma frame store, keyed by video_id)
  -> EasyOCR extraction
```

Orchestration lives in `src/ingest.py`. Speech code is in `src/speech.py`, frame extraction in `src/frame_extractor.py`, OCR in `src/ocr.py`.

All artifacts are keyed by `video_id` (sanitized filename stem or YouTube ID):
- `data/transcripts/{video_id}.json`
- `data/transcripts_chunks/{video_id}.json`
- `data/frames/{video_id}/`
- `data/chroma_db/text/{video_id}`
- `data/chroma_db/frames/{video_id}`

Frame documents are linked to transcript chunks via `chunk_id` metadata, enabling frame retrieval at query time.

---

## Retrieval & Chat Pipeline

```text
User query
  -> History-aware standalone question (contextualize_qa_prompt)
  -> BM25 retrieval
  -> Chroma vector retrieval
  -> Ensemble retrieval (BM25 + vector, configurable weights)
  -> Cross-encoder reranking
  -> Empty-result guard (returns a safe response if no chunks found)
  -> Related frame lookup (CLIP similarity over chunk_ids)
  -> OCR context extraction
  -> Vision decision (LLM decides if images are needed)
  -> Multimodal message assembly
  -> Structured LLM answer (response, timestamps, source, key takeaways)
  -> Memory update (sliding window + LLM-generated running summary)
```

Retrieval logic is in `src/retrieval.py`. The LangChain LCEL pipeline (`lcel_chat`) and a plain-Python fallback (`chat`) both live in `src/chat.py`. Prompt templates are in `src/prompts.py`.

**Empty retrieval guard:** if retrieval + reranking returns no documents (e.g. the query has no matching transcript chunks), the pipeline short-circuits and returns a controlled message rather than crashing on an index-out-of-bounds error.

---

## Memory

Conversation memory is managed in `src/memory.py` and persisted in PostgreSQL:

- A **sliding window** of the most recent `MAX_TURNS × 2` messages is passed as chat history on every turn.
- Older messages that fall outside the window are summarised by the summary LLM and stored as a running `summary` on the session row.
- On the next turn the summary is prepended to the history as a `system` message, so the model retains context from older parts of the conversation without blowing up the context window.
- All reads and writes enforce session ownership at the database level.

---

## LLM Roles

Five LLMs with distinct roles are configured in `config.py` and constructed in `src/llm.py`:

| Role | Provider |
|---|---|
| Main answer LLM | Gemini |
| Frame-attachment decision LLM | Groq (llama-3.1-8b) |
| Conversation summary LLM | OpenRouter |
| QA-generation / eval LLM | Groq |
| Judge LLM | OpenAI-compatible endpoint (NANAROUTER) |

The main LLM returns a structured `Answer` (Pydantic model in `src/base_models.py`) with fields: `response`, `timestamps`, `source`, and `key_takeway`.

Pipeline steps are instrumented with LangSmith `@traceable` decorators; tracing is enabled by default when `LANGSMITH_API_KEY` is set.

---

## Caching

Responses are cached in Redis (`src/cache.py`) at two points in the pipeline:

| Cache key | TTL (config) | Purpose |
|---|---|---|
| `vision:{video_id}:{query}` | `VISION_DECISION_CACHE_TTL` | Skip repeated vision-decision LLM calls for the same query |
| `main_llm_answer:{video_id}:{messages_hash}` | `CHAT_ANSWER_CACHE_TTL` | Return identical answers for identical message histories instantly |

---

## Evaluation

The evaluation pipeline (`evaluation/eval_pipeline.py`) runs three stages per video:

- **`generate_qa.py`** — generates typed QA pairs (factual, reasoning, application, comparison, misconception) from transcript chunks.
- **`retrieval_eval.py` / `reranker_eval.py`** — measure Hit Rate @1/3/5 and MRR across retrieval configurations.
- **`judge_eval.py`** — LLM-as-judge scoring of end-to-end answers (correctness, completeness, faithfulness, clarity; each 1–5).

`result_aggregation.py` combines per-video results into macro/micro summaries.

---

## Results

Evaluated on **10 lecture videos** (~600 generated QA pairs) covering programming, math, physics, chemistry, and general topics.

### Retrieval (micro-average over 603 QA pairs)

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 only | 0.47 | 0.71 | 0.82 | 0.62 |
| Vector only | 0.58 | 0.66 | 0.77 | 0.65 |
| Ensemble (BM25 + vector) | 0.59 | 0.84 | 0.92 | 0.73 |
| **Ensemble + reranker** | **0.68** | **0.91** | **0.95** | **0.81** |

Key takeaways:
- BM25 and vector retrieval are complementary: BM25 is stronger at Hit@3/@5, vector is stronger at Hit@1. Combining them beats both individually.
- Cross-encoder reranking gives the largest single improvement, lifting Hit@1 from 0.59 → 0.68 and MRR from 0.73 → 0.81.

### Reranker impact (same 603 pairs)

- MRR: **0.73 → 0.81** after reranking
- Hit@1: **0.59 → 0.68**, Hit@3: **0.84 → 0.91**, Hit@5: **0.92 → 0.95**
- Rank movement: 172 questions improved, 100 worsened, 328 unchanged

### LLM-as-judge

End-to-end answers scored by an LLM judge on correctness, completeness, faithfulness, and clarity (1–5 scale each, with written reasoning per answer). Per-video score files are in `evaluation_results/judge_eval_results/`.

---

## Architecture

```text
               +------------------+
               |  FastAPI (main)  |
               +--------+---------+
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
    auth_router   video_router   chat_router
          |             |             |
          v             v             v
    auth_service  video_service  session_service
                        |        chat_service
                        v             |
                  src/ingest.py   src/chat.py
                        |             |
         +--------------+     +-------+-------+
         |              |     |               |
         v              v     v               v
    speech.py    frame_     retrieval.py   memory.py
                 extractor   (Chroma +      (PostgreSQL)
                 .py + ocr   BM25 +
                 .py         reranker)
                        |
                        v
                 database/queries/
                 (ownership-enforced)
```

---

## Future Improvements

- **Multi-video retrieval** — currently scoped to a single video per session; extend to a unified index across videos with video-level filtering.
- **Dynamic rerank k** — currently fixed `RERANK_K`; route k by query type (summary queries need more context, factual queries less).
- **Better frame extraction** — current anchor-based + CLIP dedup is a first pass; explore shot-boundary / slide-change detection for higher precision.
- **OCR upgrade** — replace EasyOCR with a stronger extractor (e.g. PaddleOCR or a VLM-based approach) for handwritten / math-heavy slides.
- **Local judge model** — replace the API-based judge with a locally hosted model (e.g. Prometheus) to remove API dependency and cost.
- **Semantic chunking** — replace fixed-token chunking with semantic / embedding-based chunking for better topic-aligned chunks.
- **Performance optimisation** — speed up preprocessing and inference through parallel processing and model optimisation.

---

## Logging

Logging is configured in `src/logger.py`. Logs are written to both the console and:

```text
outputs/logs/app.log
```
