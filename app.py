from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
import argparse
import config
import os
from evaluation.eval_piepline import evaluation_pipeline
from evaluation.generate_qa import sample_and_generate
from src.retrieval import create_text_retriever, create_bm25_retriever, create_ensemble_retriever
from src.chat import lcel_chat
from src.ingest import create_text_docs, preprocess_video
from src.llm import eval_llm
from src.loader import load_clip_model, load_text_embedding_model, load_reranker
from src.logger import get_logger
from src.memory import chat_history
from src.utils import load_json

logger = get_logger(__name__)


def load_existing_artifacts(video_id):
    """Load persisted databases and initialize retrieval."""
    docs_path=os.path.join(str(config.TRANSCRIPTS_CHUNK_DIR),f"{video_id}.json")
    docs=load_json(docs_path)
    docs=[Document(**item) for item in docs]
    text_embedding_model=load_text_embedding_model(config.TEXT_EMBEDDING_MODEL)
    text_db_path=os.path.join(str(config.TEXT_DB_PATH), video_id)
    text_db=Chroma.from_documents(
        documents=docs,
        embedding=text_embedding_model,
        persist_directory=text_db_path,
    )
    
    frame_db_path=os.path.join(str(config.FRAME_DB_PATH), video_id)
    client=chromadb.PersistentClient(path=frame_db_path)
    frame_db=client.get_collection(name=config.FRAME_COLLECTION_NAME)
    
    bm25_retriever=create_bm25_retriever(docs)
    text_retriever=create_text_retriever(text_db)
    ensemble_retriever=create_ensemble_retriever(bm25_retriever, text_retriever)
    reranker=load_reranker()
    clip_model=load_clip_model()
    
    retrieval_components={
        'bm25_retriever':bm25_retriever,
        'text_retriever':text_retriever,
        'ensemble_retriever':ensemble_retriever,
        'reranker':reranker,
        'frame_db':frame_db,
        'clip_model':clip_model
    }


    return  retrieval_components 


def run_chat_query(query,retrieval_components):
    logger.info("User: %s", query)

    response = lcel_chat(query,retrieval_components)

    print("\nAssistant:\n")
    print(response)

    return response


def interactive_chat(retrieval_components):
    logger.info("Interactive chat started.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            logger.info("Chat ended.")
            break

        try:
            run_chat_query(query, retrieval_components)
        except KeyboardInterrupt:
            print()
            break
        except Exception:
            logger.exception("Chat failed.")


def run_eval_pipeline(video_id):
    logger.info("Starting evaluation pipeline.")

    evaluation_pipeline(video_id)
    
    logger.info("Evaluation pipeline completed.")
    
    return
    


def main(args):

    text_docs = None
    retrieval_components = None
    video_id=config.VIDEO_ID
    if args.preprocess:
        logger.info("Starting preprocessing.")

        results = preprocess_video()

        text_docs = results["text_docs"]
        video_id = results["video_id"]


        logger.info("Preprocessing completed.")

    if args.chat or args.query:

        if retrieval_components is None:
            logger.info("Loading persisted retrieval.")
            retrieval_components = load_existing_artifacts(video_id)
    

    if args.chat:
        interactive_chat(retrieval_components)

    if args.query:
        run_chat_query(args.query, retrieval_components)

    if args.eval:
        run_eval_pipeline(config.VIDEO_ID)


    if not any(
        [
            args.preprocess,
            args.chat,
            args.query,
            args.eval,
        ]
    ):
        logger.info(
            "No action specified. Use --preprocess, --chat, --query or --eval."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Educational Video Assistant"
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess videos and rebuild vector databases.",
    )

    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive chat session.",
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Ask a single question.",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run retrieval and answer evaluation.",
    )

    args = parser.parse_args()

    main(args)