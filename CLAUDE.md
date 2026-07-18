# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Video Assistant: a retrieval-augmented assistant for lecture videos. It ingests a video (YouTube URL or local file), transcribes speech with faster-whisper, extracts changed frames, runs EasyOCR, stores transcript chunks and frame embeddings in Chroma, and answers questions via a LangChain LCEL pipeline. The original notebook (`notebooks/VideoAssistant.ipynb`) is the reference implementation and is kept unchanged.

## Commands

```bash
pip install -r requirements.txt

python app.py --preprocess          # ingest video → transcripts, frames, Chroma DBs
python app.py --chat                # interactive chat loop
python app.py --query "question"    # single question
python app.py --eval                # run the evaluation pipeline
python app.py --url <youtube-url> --preprocess   # ingest a specific YouTube URL
python app.py --url <youtube-url> --chat         # ingest then chat (no --preprocess needed)

python -m evaluation.result_aggregation   # aggregate per-video eval results into summaries
```

`--url` overrides the config placeholders: with `--chat`/`--query` it triggers ingestion first so the video_id is known. Without `--url`, running `--chat`/`--query`/`--eval` without `--preprocess` requires setting `config.VIDEO_ID` and `config.VIDEO_PATH` — they are literal placeholder strings in `config.py` that must be edited (they are not env vars).

The `Dockerfile` (python:3.11-slim) uses `ENTRYPOINT ["python","app.py"]` with `CMD ["--chat"]` — `docker run -it <image>` starts the chat; append flags to override (e.g. `docker run -it <image> --query "question"`). `.dockerignore` excludes `data/`, `notebooks/`, `imp_point/`, and `.env`, so pass keys via `--env-file` and mount `data/` to persist artifacts.

There are no tests or linter configured.

API keys are loaded from `.env` at the repo root (see `config.py`): `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, optional `LANGSMITH_*` (tracing is on by default via `langsmith @traceable` decorators throughout), and `NANAROUTER_API_KEY`/`NANAROUTER_BASE_URL` (used as an OpenAI-compatible endpoint for the judge LLM).

## Architecture

`config.py` at the repo root is the single source of truth for all paths, model names, and tuning knobs (chunk sizes, retriever k values, ensemble weights, thresholds). All modules import it directly (`import config`). Everything under `data/`, `outputs/`, and generated media is gitignored; `evaluation_results/` is committed. `imp_point/` holds scratch notes (review notes, resume points), not code.

All artifacts are keyed by `video_id` (sanitized filename stem or YouTube ID): `data/transcripts/{video_id}.json`, `data/transcripts_chunks/{video_id}.json`, `data/frames/{video_id}/`, `data/chroma_db/text/{video_id}`, `data/chroma_db/frames/{video_id}`.

### Ingestion (`src/ingest.py` orchestrates)

video → `src/speech.py` (download via yt-dlp, audio extraction, chunking, Whisper transcription) → `create_text_docs` (char-threshold chunking with timestamp metadata) → text Chroma DB (BGE embeddings via `src/loader.py`) → `src/frame_extractor.py` (calibrated change detection + CLIP-similarity dedup) → `src/ocr.py` (EasyOCR) → frame Chroma DB (raw chromadb client, CLIP embeddings, cosine space).

Frame docs are linked to transcript chunks via `chunk_id` metadata (frame timestamp matched against chunk `end_sec`). This linkage is what makes frame retrieval work at query time.

### Chat/Retrieval (`src/chat.py`, `src/retrieval.py`)

Two parallel implementations of the same pipeline exist in `src/chat.py`: `chat()` (plain Python) and `lcel_chat()` (LCEL runnables) — `app.py` uses `lcel_chat`. Keep them in sync if you modify the pipeline. Flow:

1. History-aware query contextualization (only if `chat_history` is non-empty)
2. Ensemble retrieval (BM25 + Chroma vector, weights in `config.ENSEMBLE_WEIGHTS`)
3. Cross-encoder reranking (`BAAI/bge-reranker-base`)
4. Frame retrieval: CLIP-encode the query, query the frame Chroma DB filtered by the reranked docs' `chunk_id`s
5. OCR context built from frame metadata (token-budgeted, dedup via similarity)
6. A decision LLM (Groq, structured `Decision` output) decides whether to attach base64 frame images to the message
7. Main LLM (Gemini) returns a structured `Answer` (pydantic model in `src/base_models.py`: response, timestamps, source, key takeaways)

Conversation memory (`src/memory.py`) uses module-level globals: a sliding window of `MAX_TURNS*2` messages plus an LLM-generated running summary of evicted messages (`clear_memory()` resets both).

### LLM roles (`src/llm.py`)

Five LLMs with distinct roles, all configured in `config.py`: main answer LLM (Gemini), decision LLM (Groq llama-3.1-8b), summary LLM (OpenRouter), eval LLM (Groq), judge LLM (OpenAI-compatible via NANAROUTER). `build_multimodal_message` formats base64 images per provider ("gemini" vs "openrouter" content shapes).

### Evaluation (`evaluation/`)

`evaluation/eval_pipeline.py` orchestrates: generate/load QA pairs (`generate_qa.py`, structured output, typed by difficulty/question_type, saved under `evaluation_results/qa_pairs/all_qa_pairs/{video_id}.json` plus per-type subdirectories; if the all_qa_pairs file already exists it is reused, not regenerated) → retrieval eval across configs bm25/vector/ensemble/ensemble+rerank (`retrieval_eval.py`, hit rate @1/3/5 + MRR) → reranker before/after eval (`reranker_eval.py`) → LLM-as-judge answer scoring (`judge_eval.py`, scores correctness/completeness/faithfulness/clarity). Per-video JSON results land in `evaluation_results/*/{video_id}.json`; `result_aggregation.py` (a script, runs on import) computes macro/micro averages across videos into the summary JSONs.

QA generation and judge scoring deliberately `time.sleep()` between LLM calls (5s and 12s) to stay under provider rate limits — eval runs are slow by design; don't remove the sleeps.

## Conventions

- Logging via `src/logger.py` `get_logger(__name__)` — writes to console and `outputs/logs/app.log`. Use `logger`, not `print` (except direct user-facing chat output).
- LangSmith `@traceable` decorators and `.with_config({"run_name": ...})` are used on pipeline steps for tracing — preserve them when refactoring.
- Structured LLM outputs go through pydantic models in `src/base_models.py` (app) and `evaluation/base_model.py` (eval) with `.with_structured_output()`.
- Code style: minimal spacing around `=` in assignments, 2-space indent in some modules — match the file you're editing.
- `torch` is deliberately not in `requirements.txt` (it comes in transitively via sentence-transformers/EasyOCR) — don't re-add it. Models in `src/loader.py` are loaded with library defaults (no explicit device); `config.DEVICE` is currently unused. Don't introduce GPU-conditional loading without checking the Docker/CPU deployment path.
