import random
import os
import math
from langsmith import traceable

import config
import time
from src.utils import save_json
from evaluation import prompts
from evaluation import base_model
from src.logger import get_logger


logger=get_logger(__name__)


@traceable(name="generate_qa_pairs")
def generate_qa_pairs(docs,qa_type,eval_llm,target_n):
    qa_pairs=[]

    structured_eval_llm=eval_llm.with_structured_output(qa_type["base"])
    chain=(qa_type["prompt"]|structured_eval_llm).with_config(run_name=qa_type["type"])
    
    doc_pool=docs.copy()
    random.shuffle(doc_pool)
    
    max_per_doc=math.ceil(target_n/len(doc_pool))  #bec if we have 10 docs and we need 20 pairs, we need to generate 2 pairs per doc.
    max_per_doc=min(max_per_doc,6)  #max 6 pairs per doc to avoid overfitting to a single chunk.
    max_per_doc=max(max_per_doc,2)  #min 2 pairs per doc to ensure we get enough pairs.
    
    doc_tried=0
    for doc in doc_pool:
        if len(qa_pairs)>=target_n :
            break

        doc_tried+=1
        response=chain.invoke({"content":doc.page_content,"n":max_per_doc})
        
        #Assuming The chunk that generated the question is the ground-truth chunk.(Not perfect but simplify)

        for pair in response.pairs:
            pair_dict=pair.model_dump()
            qa_pairs.append({  
                **pair_dict,
                "start": doc.metadata["start"],
                "end": doc.metadata["end"],
                "chunk_id": doc.metadata["chunk_id"],
                "video_id": doc.metadata["video_id"],
                "content": doc.page_content,
            })
        
    time.sleep(5)  # Sleep for 5 seconds to avoid rate limiting
    return qa_pairs


def sample_and_generate(docs,eval_llm,video_id,n=config.TOTAL_QA_PAIRS):
    logger.info(f"Sampling documents for QA generation. Total docs: {len(docs)}")
    eval_docs=random.sample(docs,min(len(docs),2*n))
    logger.info(f"Sampled {len(eval_docs)} documents for QA generation.")
    
    qa_pairs_type = {
            "Factual QA": {
                "base": base_model.FactualQA_List,
                "prompt": prompts.factual_qa_generation_prompt,
                "ratio": 0.40,
            },
            "Comparison QA": {
                "base": base_model.ComparisonQA_List,
                "prompt": prompts.comparison_qa_generation_prompt,
                "ratio": 0.20,
            },
            "Reasoning QA": {
                "base": base_model.ReasoningQA_List,
                "prompt": prompts.reasoning_qa_generation_prompt,
                "ratio": 0.15,
            },
            "Application QA": {
                "base": base_model.ApplicationQA_List,
                "prompt": prompts.application_qa_generation_prompt,
                "ratio": 0.15,
            },
            "Misconception QA": {
                "base": base_model.MisconceptionQA_List,
                "prompt": prompts.misconception_qa_generation_prompt,
                "ratio": 0.10,
            },
        }
  
  
    all_pairs=[]
    
    for qa_type,info in qa_pairs_type.items():
        target=math.ceil(n*info["ratio"])
        
        
        logger.info(f"Generating approximately {target} {qa_type} for evaluation.")
        qa_pairs=generate_qa_pairs(eval_docs,{"base":info["base"],"prompt":info["prompt"],"type":qa_type},eval_llm,target_n=target)
        logger.info(f"Generated {len(qa_pairs)} {qa_type} for evaluation.")
        
        all_pairs.extend(qa_pairs)
        
        path=os.path.join(config.QA_PAIRS_DIR,f"{qa_type}")
        os.makedirs(path, exist_ok=True)
        save_json(qa_pairs, os.path.join(path,f"{video_id}.json"))
        
    
    os.makedirs(os.path.join(config.QA_PAIRS_DIR,"all_qa_pairs"), exist_ok=True)
    save_json(all_pairs, os.path.join(config.QA_PAIRS_DIR,"all_qa_pairs",f"{video_id}.json"))
      
    return all_pairs
