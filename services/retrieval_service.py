from langchain_chroma import Chroma
from langchain_core.documents import Document

import chromadb
import config
import os

from api.exceptions import not_found
from src.retrieval import create_text_retriever,create_bm25_retriever,create_ensemble_retriever
from src.loader import load_clip_model,load_text_embedding_model,load_reranker

from src.utils import load_json

from collections import OrderedDict
from threading import Lock


 
retrieval_cache = OrderedDict()
retrieval_cache_lock = Lock()
video_load_locks = {}
video_load_locks_guard = Lock()
 
  
 
def get_video_load_lock(video_id):
    with video_load_locks_guard:
        lock = video_load_locks.get(video_id)
        if lock is None:
            lock = Lock()
            video_load_locks[video_id] = lock
        return lock
 
 
def get_cached_retrieval_components(video_id):
    with retrieval_cache_lock:
        if video_id not in retrieval_cache:
            return None
        retrieval_cache.move_to_end(video_id)
        return retrieval_cache[video_id]
 
 
def set_cached_retrieval_components(video_id, components):
    maxsize = config.RETRIEVAL_CACHE_MAXSIZE
    with retrieval_cache_lock:
        if video_id in retrieval_cache:
            retrieval_cache.move_to_end(video_id)
            retrieval_cache[video_id] = components
            return
        retrieval_cache[video_id] = components
        while len(retrieval_cache) > maxsize:
            retrieval_cache.popitem(last=False)
 
 
def invalidate_retrieval_cache(video_id=None):
    with retrieval_cache_lock:
        if video_id is None:
            retrieval_cache.clear()
            return
        retrieval_cache.pop(video_id, None)



def load_retrieval_components(video_id):
    cached = get_cached_retrieval_components(video_id)
    if cached is not None:
        return cached

    with get_video_load_lock(video_id):
        cached = get_cached_retrieval_components(video_id)
        if cached is not None:
            return cached

        components = build_retrieval_components(video_id)
        set_cached_retrieval_components(video_id, components)
        return components
 
 
def build_retrieval_components(video_id):
    
    docs_path=os.path.join(str(config.TRANSCRIPTS_CHUNK_DIR),f"{video_id}.json")
    if not os.path.exists(docs_path):
        raise not_found(f"Transcripts for video {video_id} not found. Please run the ingestion process first.")
    
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
