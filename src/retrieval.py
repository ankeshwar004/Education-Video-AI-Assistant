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





def rerank(query,docs,reranker,k=config.RERANK_K):

  pairs=[(query,doc.page_content) for doc in docs]

  scores=reranker.predict(pairs)
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



@traceable(name="Frame Retriever")
def frame_retriever(query,chunk_ids,frame_db,clip_model,n=config.FRAME_RETRIEVER_N):

  query_embedding=clip_model.encode(query,convert_to_numpy=True,normalize_embeddings=True)

  return frame_db.query(
      query_embeddings=[query_embedding.tolist()],
      where={"chunk_id": {"$in": chunk_ids}},
      n_results=n,
      include=["metadatas","distances"]
      )


@traceable(name="Docs Retriever")
def docs_retriever(query,retriever):
  return retriever.invoke(query)
