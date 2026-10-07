import config
import easyocr

from langchain_huggingface import HuggingFaceEmbeddings
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from functools import lru_cache


def load_whisper_model(model_name=config.WHISPER_MODEL):
    return WhisperModel(model_name)

def load_reader():
    return easyocr.Reader(['en'])

def load_clip_model(model_name=config.CLIP_MODEL):
    return _load_clip_model(model_name)

def load_text_embedding_model(model_name=config.TEXT_EMBEDDING_MODEL):
    return _load_text_embedding_model(model_name)

def load_reranker(model_name=config.RERANKER_MODEL):
    return _load_reranker(model_name)


@lru_cache(maxsize=4)
def _load_clip_model(model_name):
    return SentenceTransformer(model_name)
 
@lru_cache(maxsize=4)
def _load_text_embedding_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)
 
@lru_cache(maxsize=4)
def _load_reranker(model_name):
    return CrossEncoder(model_name)
