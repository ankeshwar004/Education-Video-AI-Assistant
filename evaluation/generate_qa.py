import random
import os
from pydantic import BaseModel
from langsmith import traceable

import config
from src.utils import save_json
from evaluation.prompts import qa_generation_prompt
from src.logger import get_logger


logger=get_logger(__name__)


class QAPairs(BaseModel):
  question: str
  answer: str


class QAPairsList(BaseModel):
  pairs: list[QAPairs]

@traceable(name="generate_qa_pairs")
def generate_qa_pairs(docs,eval_llm,n):
  qa_pairs=[]

  structured_eval_llm=eval_llm.with_structured_output(QAPairsList)
  for doc in docs:
    chain=(qa_generation_prompt|structured_eval_llm).with_config(run_name="QA_Pairs")

    response=chain.invoke({"content":doc.page_content,"n":n})

    #Assuming The chunk that generated the question is the ground-truth chunk.(Not perfect but simplify)

    for pair in response.pairs:
      qa_pairs.append({
      "question": pair.question,
      "answer": pair.answer,
      "start": doc.metadata["start"],
      "end": doc.metadata["end"],
      "chunk_id": doc.metadata["chunk_id"],
      "video_id": doc.metadata["video_id"],
      "content": doc.page_content
      })
  return qa_pairs


def sample_and_generate(docs,eval_llm,video_id,n=config.EVAL_QA_PAIRS_PER_CHUNK):
    eval_docs=random.sample(docs, min(50,len(docs)//2))
    logger.info(f"Sampled {len(eval_docs)} documents for QA generation.")
    
    qa_pairs=generate_qa_pairs(eval_docs,eval_llm,n)
    logger.info(f"Generated {len(qa_pairs)} QA pairs from sampled documents.")
    
    os.makedirs(config.QA_PAIRS_DIR, exist_ok=True)
    save_json(qa_pairs, os.path.join(config.QA_PAIRS_DIR,f"{video_id}.json"))
    return qa_pairs
