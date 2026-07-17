import config
import easyocr

from langchain_huggingface import HuggingFaceEmbeddings
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


def load_whisper_model(model_name=config.WHISPER_MODEL):
    return WhisperModel(model_name)

def load_reader():
    return easyocr.Reader(['en'])

def load_clip_model(model_name=config.CLIP_MODEL):
    return SentenceTransformer(model_name)

def load_text_embedding_model(model_name=config.TEXT_EMBEDDING_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_reranker(model_name=config.RERANKER_MODEL):
    return CrossEncoder(model_name)