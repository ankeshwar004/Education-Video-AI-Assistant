from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

import config
from src.logger import get_logger
from utils import is_similar,token_count


logger = get_logger(__name__)


def create_text_retriever(text_db):
    return text_db.as_retriever(
        search_type='similarity',
        search_kwargs={'k':config.TEXT_RETRIEVER_K}
    )


def create_bm25_retriever(text_docs):
    bm25_retrieve =BM25Retriever.from_documents(text_docs)
    bm25_retriever.k = config.BM25_K
    return bm25_retriever


def create_ensemble_retriever(bm25_retriever, text_retriever):
    logger.info("Creating ensemble retriever...")
    return EnsembleRetriever(
        retrievers=[bm25_retriever,text_retriever],
        weights=config.ENSEMBLE_WEIGHTS)


def load_reranker(model_name=config.RERANKER_MODEL):
    
    return CrossEncoder(model_name)


def rerank(query,docs,reranker,k=5):

  pairs=[(query,doc.page_content) for doc in docs]

  scores=reranker.predict(pairs)
  reranked=sorted(zip(docs,scores),key=lambda x:x[1],reverse=True)

  return [doc for doc,score in reranked[:k]]


def build_text_context(docs):
  formatted=[]

  for doc in docs:
      context= f"""
      Timestamp: {doc.metadata['start']} - {doc.metadata['end']}
      Content: {doc.page_content}
      """
      formatted.append(context)

  return "\n\n".join(formatted)


def build_ocr_context(frames_metadata,max_token=config.MAX_OCR_TOKEN):

  if not frames_metadata:
    return "No relevant on-screen text found for this query"

  ocr,seen=[],[]
  total_token=0

  for frame in frames_metadata:
    ocr_text=frame['ocr_text']

    if not ocr_text or is_similar(ocr_text,seen):
      continue

    block=f"Frame: {frame['frame_ts']}\nOCR Text: {ocr_text}"
    block_token=token_count(block)

    if total_token+block_token>max_token:
      break

    ocr.append(block)
    seen.append(ocr_text)
    total_token+=block_token

  return "\n\n".join(ocr)





def initialize_retrieval(text_docs, text_db):
    text_retriever=create_text_retriever(text_db)
    bm25_retriever=create_bm25_retriever(text_docs)
    ensemble_retriever=create_ensemble_retriever(bm25_retriever, text_retriever)
    reranker=load_reranker(config.RERANKER_MODEL)
    logger.info("Retrieval initialized.")
    return {
        "text_retriever": text_retriever,
        "bm25_retriever": bm25_retriever,
        "ensemble_retriever": ensemble_retriever,
        "reranker": reranker,
    }
