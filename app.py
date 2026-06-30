
import argparse

import config
from src.chat import initialize_chat_pipeline
from src.ingest import preprocess_video
from src.logger import get_logger
from src.retrieval import initialize_retrieval


logger = get_logger(__name__)


def build_app(url=config.YOUTUBE_URL, video_path=None):
    config.ensure_directories()
    state=preprocess_video(url=url, video_path=video_path)
    retrieval_components=initialize_retrieval(state["text_docs"], state["text_db"])
    chat_fn=initialize_chat_pipeline(
        retrieval_components,
        state["clip_model"],
        state["frame_db"]
    )
    return chat_fn


def run_cli(chat_fn):
    """Run a simple terminal chat interface."""
    print("Video Assistant is ready. Type 'exit' to quit.")
    while True:
        query=input("\nYou: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        result=chat_fn(query)
        print("\nAssistant:")
        print(result)


def main():
    parser=argparse.ArgumentParser(description="Run the Video Assistant.")
    parser.add_argument("--url", default=config.YOUTUBE_URL)
    parser.add_argument("--video-path", default=None)
    args=parser.parse_args()

    logger.info("Initializing application...")
    chat_fn=build_app(url=args.url, video_path=args.video_path)
    run_cli(chat_fn)


if __name__ == "__main__":
    main()
