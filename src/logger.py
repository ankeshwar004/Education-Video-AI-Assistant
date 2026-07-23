import logging
from config import LOG_DIR


def get_logger(name="VideoAssistant"):
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(LOG_DIR/"app.log"),
                  logging.StreamHandler()]
    )
    return logging.getLogger(name)
