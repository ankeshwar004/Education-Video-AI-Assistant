# Video Assistant

Video Assistant is a retrieval-augmented assistant for lecture videos. It downloads or accepts a video, extracts audio, transcribes speech with Whisper, extracts important frames, runs OCR, stores transcript and frame evidence in Chroma, and answers student questions using a LangChain retrieval and LLM pipeline.

The original notebook remains unchanged in `notebooks/VideoAssistant.ipynb`.

## Folder Structure

```text
RAG_Video_Assistant/
|-- app.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- Dockerfile
|-- .dockerignore
|-- .env
|-- data/
|   |-- videos/
|   |-- audio/
|   |-- transcripts/
|   |-- transcripts_chunks/
|   |-- frames/
|   `-- chroma_db/
|-- src/
|   |-- ingest.py
|   |-- retrieval.py
|   |-- chat.py
|   |-- memory.py
|   |-- llm.py
|   |-- prompts.py
|   |-- base_models.py
|   |-- ocr.py
|   |-- speech.py
|   |-- frame_extractor.py
|   |-- loader.py
|   |-- logger.py
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
|-- notebooks/
|   `-- VideoAssistant.ipynb
`-- outputs/
    `-- logs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All paths, model names, and tuning knobs (chunk sizes, retriever k values, ensemble weights, thresholds) live in `config.py` at the repo root — it is the single source of truth and every module imports it directly.

API keys are loaded from a `.env` file at the repo root:

```text
GEMINI_API_KEY=...        # main answer LLM (Gemini)
GROQ_API_KEY=...          # decision + eval LLMs
OPENROUTER_API_KEY=...    # summary LLM
NANAROUTER_API_KEY=...    # judge LLM (OpenAI-compatible endpoint)
NANAROUTER_BASE_URL=...
LANGSMITH_API_KEY=...     # optional; tracing is enabled by default
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=RAG_VideoAssistant
```

Before running `--chat`, `--query`, or `--eval` without `--preprocess`, edit `config.VIDEO_ID` and `config.VIDEO_PATH` in `config.py` — they are literal placeholder strings, not environment variables.

## How To Run

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

## Docker

A `Dockerfile` (python:3.11-slim) is included. The default command runs the interactive chat; override flags at run time:

```bash
docker build -t video-assistant .
docker run -it video-assistant                       # defaults to --chat
docker run --url <youtube-url> --chat
docker run --url <youtube-url> --preprocess 
```

`.dockerignore` excludes `data/`, `notebooks/`, and `.env`, so pass API keys at run time (e.g. `docker run --env-file .env ...`) and mount `data/` as a volume if you want artifacts to persist.

## Preprocessing Pipeline

```text
Video
  -> Audio extraction
  -> Audio chunking
  -> Whisper transcription
  -> Transcript document chunking
  -> Text embeddings
  -> Chroma text storage
  -> Changed-frame extraction
  -> CLIP frame embeddings
  -> EasyOCR extraction
  -> Chroma frame storage
```

The orchestration lives in `src/ingest.py`. Speech code is in `src/speech.py`, frame extraction is in `src/frame_extractor.py`, and OCR is in `src/ocr.py`.

All artifacts are keyed by `video_id` (the sanitized filename stem or YouTube ID): `data/transcripts/{video_id}.json`, `data/transcripts_chunks/{video_id}.json`, `data/frames/{video_id}/`, `data/chroma_db/text/{video_id}`, and `data/chroma_db/frames/{video_id}`. Frame documents are linked to transcript chunks via `chunk_id` metadata, which is what enables frame retrieval at query time.

## Retrieval Pipeline

```text
User query
  -> History-aware standalone question
  -> BM25 retrieval
  -> Chroma vector retrieval
  -> Ensemble retrieval
  -> Cross-encoder reranking
  -> Related frame lookup
  -> OCR context + optional images
  -> Structured LLM answer
```

Retrieval logic lives in `src/retrieval.py`. Prompt templates live in `src/prompts.py`. The chat pipeline lives in `src/chat.py`.

## LLM Roles

Five LLMs with distinct roles are configured in `config.py` and constructed in `src/llm.py`:

| Role | Provider |
|---|---|
| Main answer LLM | Gemini |
| Frame-attachment decision LLM | Groq (llama-3.1-8b) |
| Conversation summary LLM | OpenRouter |
| QA-generation / eval LLM | Groq |
| Judge LLM | OpenAI-compatible endpoint (NANAROUTER) |

The main LLM returns a structured `Answer` (pydantic model in `src/base_models.py`) containing the response, timestamps, source, and key takeaways. Conversation memory (`src/memory.py`) keeps a sliding window of recent turns plus an LLM-generated running summary of older, evicted messages.

Pipeline steps are instrumented with LangSmith `@traceable` decorators; tracing is on by default when `LANGSMITH_API_KEY` is set.

## Evaluation

The evaluation pipeline (`evaluation/eval_pipeline.py`) runs three stages per video:

- `evaluation/generate_qa.py` — generates typed QA pairs (factual, reasoning, application, comparison, misconception) from transcript chunks
- `evaluation/retrieval_eval.py` and `evaluation/reranker_eval.py` — measure Hit Rate @1/3/5 and MRR across retrieval configurations
- `evaluation/judge_eval.py` — LLM-as-judge scoring of end-to-end answers (correctness, completeness, faithfulness, clarity, each 1–5)

`evaluation/result_aggregation.py` combines per-video results into macro/micro summaries.

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

- BM25 and vector retrieval are complementary: BM25 is stronger at Hit@3/@5, vector is stronger at Hit@1. Combining them in an ensemble beats both individually.
- Cross-encoder reranking gives the largest single improvement, lifting Hit@1 from 0.59 to 0.68 and MRR from 0.73 to 0.80.

### Reranker impact (same 603 pairs)

- MRR: **0.73 → 0.81** after reranking
- Hit@1: **0.59 → 0.68**, Hit@3: **0.84 → 0.91**, Hit@5: **0.92 → 0.95**
- Rank movement: 172 questions improved, 100 worsened, 328 unchanged

Note: the post-rerank MRR here (0.81) is measured over the full ~20-candidate list, while the ensemble+reranker row in the table above scored a run truncated to the top 5 (0.80) — gold chunks landing at rank 6+ count as a miss there. Hit rates @1/3/5 are unaffected and match exactly.

### LLM-as-judge

End-to-end answers are also scored by an LLM judge on correctness, completeness, faithfulness, and clarity (1–5 scale each, with written reasoning per answer). Per-video score files are stored in `evaluation_results/judge_eval_results/`.

## Architecture

```text
              +----------------+
              |    app.py      |
              +-------+--------+
                      |
                      v
              +-------+--------+
              |  src/ingest.py |
              +-------+--------+
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
  speech.py   frame_extractor.py   ocr.py
        |             |             |
        +-------------+-------------+
                      |
                      v
              +-------+--------+
              | Chroma stores  |
              +-------+--------+
                      |
                      v
              +-------+--------+
              | retrieval.py   |
              +-------+--------+
                      |
                      v
              +-------+--------+
              |    chat.py     |
              +----------------+
```

## Future Improvements

- Multi-video retrieval — currently scoped to a single video per session; extend to a unified index across videos with video-level filtering.
- Dynamic k for reranking — currently fixed `RERANK_K`; route k by query type (summary queries need more context, factual queries need less).
- Better frame extraction — current anchor-based + CLIP dedup approach is a first pass; explore shot-boundary/slide-change detection for higher precision.
- OCR upgrade — replace EasyOCR with a stronger extractor (e.g. PaddleOCR or VLM-based) for handwritten/math-heavy slides.
- Local judge model — replace API-based Mistral Large judge with a locally hosted model (e.g. Prometheus) to remove API dependency and cost.
- Persistent chat memory — currently in-memory/local variable; move to a persistent store (SQLite/Redis).
- Semantic chunking — replace fixed-token chunking with semantic/embedding-based chunking for better topic-aligned chunks.
-Performance optimization – Speed up preprocessing and inference through caching, parallel processing, and model optimization techniques.
- Streamlit UI — wrap the chat pipeline in a simple web interface for demo purposes.


## Logging

Logging is configured in `src/logger.py`. Logs are written to both the console and:

```text
outputs/logs/app.log
```
