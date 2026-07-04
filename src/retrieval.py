import chromadb
import tiktoken
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

import config
from src.logger import get_logger
from src.utils import is_similar,token_count
from src.loader import load_reranker


logger = get_logger(__name__)


text_retriever=None
bm25_retriever=None
ensemble_retriever=None
reranker=None
frame_db=None
clip_model=None


def create_text_retriever(text_db):
    return text_db.as_retriever(
        search_type='similarity',
        search_kwargs={'k':config.TEXT_RETRIEVER_K}
    )


def create_bm25_retriever(text_docs):
    bm25_retriever=BM25Retriever.from_documents(text_docs)
    bm25_retriever.k=config.BM25_K
    return bm25_retriever


def create_ensemble_retriever(bm25_retriever, text_retriever):
    logger.info("Creating ensemble retriever...")
    return EnsembleRetriever(
        retrievers=[bm25_retriever,text_retriever],
        weights=config.ENSEMBLE_WEIGHTS)





def rerank(query,docs,reranker_arg=None,k=5):

  active_reranker=reranker_arg if reranker_arg is not None else reranker
  if active_reranker is None:
    raise RuntimeError("Retrieval has not been initialized with a reranker")

  pairs=[(query,doc.page_content) for doc in docs]

  scores=active_reranker.predict(pairs)
  reranked=sorted(zip(docs,scores),key=lambda x:x[1],reverse=True)

  return [doc for doc,score in reranked[:k]]

@traceable(name="Build Text Context")
def build_text_context(docs):
  formatted=[]

  for doc in docs:
      context= f"""
      Timestamp: {doc.metadata['start']} - {doc.metadata['end']}
      Content: {doc.page_content}
      """
      formatted.append(context)

  return "\n\n".join(formatted)


@traceable(name="Build OCR Context")
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




def initialize_retrieval(text_docs, text_db, frame_db_arg=None, clip_model_arg=None):
    global text_retriever, bm25_retriever, ensemble_retriever, reranker, frame_db, clip_model

    text_retriever=create_text_retriever(text_db)
    bm25_retriever=create_bm25_retriever(text_docs)
    ensemble_retriever=create_ensemble_retriever(bm25_retriever, text_retriever)
    reranker=load_reranker(config.RERANKER_MODEL)
    frame_db=frame_db_arg
    clip_model=clip_model_arg
    logger.info("Retrieval initialized.")
    return {
        "text_retriever": text_retriever,
        "bm25_retriever": bm25_retriever,
        "ensemble_retriever": ensemble_retriever,
        "reranker": reranker,
    }


@traceable(name="Frame Retriever")
def frame_retriever(query,chunk_ids,n=config.FRAME_RETRIEVER_N):

  if frame_db is None or clip_model is None:
    raise RuntimeError("Retrieval has not been initialized with frame_db and clip_model")

  query_embedding=clip_model.encode(query,convert_to_numpy=True,normalize_embeddings=True)

  return frame_db.query(
      query_embeddings=[query_embedding.tolist()],
      where={"chunk_id": {"$in": chunk_ids}},
      n_results=n,
      include=["metadatas","distances"]
      )
