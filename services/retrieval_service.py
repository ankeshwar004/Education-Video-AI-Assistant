from langchain_chroma import Chroma
from langchain_core.documents import Document

import chromadb
import config
import os

from src.retrieval import create_text_retriever,create_bm25_retriever,create_ensemble_retriever
from src.loader import load_clip_model,load_text_embedding_model,load_reranker

from src.utils import load_json


def load_retrieval_components(video_id):
    
    docs_path=os.path.join(str(config.TRANSCRIPTS_CHUNK_DIR),f"{video_id}.json")
    docs=load_json(docs_path)
    docs=[Document(**item) for item in docs]
    
    text_embedding_model=load_text_embedding_model(config.TEXT_EMBEDDING_MODEL)
    
    text_db_path=os.path.join(str(config.TEXT_DB_PATH), video_id)
    text_db=Chroma(
        persist_directory=text_db_path,
        embedding_function=text_embedding_model,
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




# For later add ->threathing
# def get_retrieval_components(video_id):
#     with _lock:
#         if video_id not in _store:
#             logger.info("Loading retrieval artifacts for %s", video_id)
#             _store[video_id] = load_existing_artifacts(video_id)
#         return _store[video_id]
 
 
# def invalidate_retrieval_components(video_id):
#     with _lock:
#         _store.pop(video_id, None)