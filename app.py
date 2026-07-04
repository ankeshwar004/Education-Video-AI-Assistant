
import argparse
import config
from src.chat import chat
from src.ingest import preprocess_video
from src.logger import get_logger
from src.retrieval import initialize_retrieval


logger = get_logger(__name__)


def main(run_preprocess: bool = False):
	if run_preprocess:
		logger.info("Starting preprocessing...")
		results = preprocess_video()

		initialize_retrieval(results["text_docs"], results["text_db"], frame_db_arg=results.get("frame_db"), clip_model_arg=results.get("clip_model"))

		logger.info("Preprocessing and retrieval initialization complete.")
	else:
		logger.info("No preprocessing requested. Use --preprocess to run ingestion.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--preprocess", action="store_true", help="Run video preprocessing and initialize retrieval")
	args = parser.parse_args()

	main(run_preprocess=args.preprocess)

