from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langsmith import traceable

import chromadb
import config
import os
from evaluation.generate_qa import sample_and_generate
from evaluation.retrieval_eval import evaluate_retrieval_configs
from evaluation.reranker_eval import evaluate_reranker
from evaluation.judge_eval import evaluate_llm_as_judge
from src.chat import lcel_chat
from src.retrieval import create_text_retriever, create_bm25_retriever, create_ensemble_retriever, rerank
from src.loader import load_text_embedding_model, load_reranker , load_clip_model
from src.logger import get_logger
from src.utils import load_json


logger=get_logger(__name__)

def get_eval_llm():
    return ChatGroq(model=config.EVAL_LLM_MODEL)

def get_judge_llm():
    return ChatOpenAI(model=config.JUDGE_LLM_MODEL,base_url=config.OPENAI_BASE_URL,api_key=config.OPENAI_API_KEY)

@traceable(name="Evaluation Pipeline", run_type="chain")
def evaluation_pipeline(video_id):
    
    eval_llm=get_eval_llm()
    logger.info("Starting evaluation pipeline...")
    
    logger.info("Loading Docs")
    docs_path=os.path.join(str(config.TRANSCRIPTS_CHUNK_DIR),f"{video_id}.json")
    docs=load_json(docs_path)
    docs=[Document(**item) for item in docs]
    
    qa_pairs_path=os.path.join(str(config.QA_PAIRS_DIR),"all_qa_pairs",f"{video_id}.json")
    if os.path.exists(qa_pairs_path):
        logger.info(f"Loading existing QA pairs for {video_id}")
        qa_pairs=load_json(qa_pairs_path)
    else:   
        logger.info(f"Generating QA pairs for evaluation.")    
        qa_pairs=sample_and_generate(docs,eval_llm,video_id,n=config.TOTAL_QA_PAIRS)
        
    
    logger.info("Loading  DB and initializing retrieval components...")
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
    
    retrieval={
        'bm25_retriever':bm25_retriever,
        'text_retriever':text_retriever,
        'ensemble_retriever':ensemble_retriever,
        'reranker':reranker,
        'frame_db':frame_db,
        'clip_model':clip_model
    }
    
    
    logger.info(f"Retrieval evaluation Started.")
    retrieval_eval=evaluate_retrieval_configs(qa_pairs,retrieval,eval_llm,video_id)
    
    logger.info(f"Reranker evaluation Started.")
    reranker_eval=evaluate_reranker(qa_pairs,retrieval,video_id)
    
    judge_llm=get_judge_llm()
    logger.info(f"LLM as Judge evaluation Started.")
    llm_as_judge_eval=evaluate_llm_as_judge(qa_pairs,lcel_chat,retrieval,judge_llm,video_id)
    