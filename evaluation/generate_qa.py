import random
import os
from langsmith import traceable

import config
from src.utils import save_json
from evaluation.prompts import general_qa_generation_prompt,misconception_qa_generation_prompt
from evaluation.base_model import GeneralQA_List,MisconceptionQA_List
from src.logger import get_logger


logger=get_logger(__name__)


@traceable(name="generate_qa_pairs")
def generate_qa_pairs(docs,qa_type,eval_llm,n):
  qa_pairs=[]

  structured_eval_llm=eval_llm.with_structured_output(qa_type["base"])
  for doc in docs:
    chain=(qa_type["prompt"]|structured_eval_llm).with_config(run_name=qa_type["type"])

    response=chain.invoke({"content":doc.page_content,"n":n})
    
    question_type = "misconception" if qa_type["type"] == "Misconception QA" else None
    #Assuming The chunk that generated the question is the ground-truth chunk.(Not perfect but simplify)

    for pair in response.pairs:
      pair_dict=pair.model_dump()
      qa_pairs.append({
      **pair_dict,
      "question_type":question_type or pair_dict["question_type"],
      "start": doc.metadata["start"],
      "end": doc.metadata["end"],
      "chunk_id": doc.metadata["chunk_id"],
      "video_id": doc.metadata["video_id"],
      "content": doc.page_content,
      })
  return qa_pairs


def sample_and_generate(docs,eval_llm,video_id,n=config.EVAL_QA_PAIRS_PER_CHUNK):
    logger.info(f"Sampling documents for QA generation. Total docs: {len(docs)}")
    eval_docs=random.sample(docs, min(50,len(docs)//2))
    logger.info(f"Sampled {len(eval_docs)} documents for QA generation.")
    
    qa_pairs_type = {
        "General QA": {
            "base": GeneralQA_List,
            "prompt": general_qa_generation_prompt,
        },
        "Misconception QA": {
            "base": MisconceptionQA_List,
            "prompt": misconception_qa_generation_prompt,
        }}
    all_pairs=[]
    
    for qa_type,info in qa_pairs_type.items():
        logger.info(f"Generating {qa_type} for evaluation.")
        qa_pairs=generate_qa_pairs(eval_docs,{"base":info["base"],"prompt":info["prompt"],"type":qa_type},eval_llm,n)
        logger.info(f"Generated {len(qa_pairs)} {qa_type} for evaluation.")
        
        all_pairs.extend(qa_pairs)
        
        path=os.path.join(config.QA_PAIRS_DIR,f"{qa_type}")
        os.makedirs(path, exist_ok=True)
        save_json(qa_pairs, os.path.join(path,f"{video_id}.json"))
    
    os.makedirs(os.path.join(config.QA_PAIRS_DIR,"all_qa_pairs"), exist_ok=True)
    save_json(all_pairs, os.path.join(config.QA_PAIRS_DIR,"all_qa_pairs",f"{video_id}.json"))
      
    return all_pairs
