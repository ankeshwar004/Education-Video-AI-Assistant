# Video Assistant

Video Assistant is a retrieval-augmented assistant for lecture videos. It downloads or accepts a video, extracts audio, transcribes speech with Whisper, extracts important frames, runs OCR, stores transcript and frame evidence in Chroma, and answers student questions using the same LangChain retrieval and LLM pipeline from the original notebook.

The original notebook remains unchanged in `notebooks/VideoAssistant (8).ipynb`.

## Folder Structure

```text
VideoAssistant/
|-- app.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- videos/
|   |-- audio/
|   |-- transcripts/
|   |-- frames/
|   `-- chroma_db/
|-- src/
|   |-- ingest.py
|   |-- retrieval.py
|   |-- chat.py
|   |-- prompts.py
|   |-- ocr.py
|   |-- speech.py
|   |-- frame_extractor.py
|   |-- logger.py
|   `-- utils.py
|-- notebooks/
|   `-- VideoAssistant (8).ipynb
|-- evaluation/
|   |-- generate_qa.py
|   |-- retrieval_eval.py
|   `-- answer_eval.py
`-- outputs/
    |-- logs/
    `-- results/
```

## Installation

```bash
pip install -r requirements.txt
```

Set the required API keys before running:

```bash
set GEMINI_API_KEY=your_key
set GROQ_API_KEY=your_key
set OPENROUTER_API_KEY=your_key
```

On macOS or Linux, use `export` instead of `set`.

## How To Run

Run with the configured YouTube URL:

```bash
python app.py
```

Run with a local video file:

```bash
python app.py --video-path path/to/video.mp4
```

Run with a different YouTube URL:

```bash
python app.py --url "https://www.youtube.com/watch?v=..."
```

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

## Evaluation

Evaluation code from the notebook is split into:

- `evaluation/generate_qa.py` for QA pair generation
- `evaluation/retrieval_eval.py` for retrieval and reranker evaluation
- `evaluation/answer_eval.py` for answer quality judging

The evaluation logic is preserved from the notebook.

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

## Logging

Logging is configured in `src/logger.py`. Logs are written to both the console and:

```text
outputs/logs/app.log
```
