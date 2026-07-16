from pydantic import BaseModel
from langsmith import traceable

import config
import os
import time
from src.utils import save_json,load_json
from evaluation.prompts import judge_prompt
from src.memory import clear_memory
from src.logger import get_logger


logger=get_logger(__name__)

class JudgeScore(BaseModel):
    reasoning: str
    correctness: int
    completeness: int
    faithfulness: int
    clarity: int


def judge_answer(question,reference_answer,generated_answer,content,llm):

    judge_llm=llm.with_structured_output(JudgeScore)
    chain=(judge_prompt|judge_llm).with_config(run_name="judge_answer")
    response=chain.invoke({
       "question": question,
       "reference_answer": reference_answer,
       "generated_answer": generated_answer,
       "content": content
    })

    return response

@traceable(name="evaluate_llm_as_judge")
def evaluate_llm_as_judge(qa_pairs,chat_fn,retrieval,llm,video_id,force_restart=False):
    
    os.makedirs(config.JUDGE_EVAL_RESULTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.JUDGE_EVAL_RESULTS_DIR, f"{video_id}_checkpoint.json")
    final_path = os.path.join(config.JUDGE_EVAL_RESULTS_DIR, f"{video_id}.json")

    all_scores = []
    start_index = 0
    
    if not force_restart and os.path.exists(checkpoint_path):
        try:
            checkpoint_data = load_json(checkpoint_path)
            all_scores = checkpoint_data.get("scores", [])
            start_index = len(all_scores)
            logger.info(f"Resuming evaluation from index {start_index} for video {video_id}.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint for video {video_id}: {e}. Starting from scratch.")
            all_scores = []
            start_index = 0

    for idx in range(start_index, len(qa_pairs)):
        qa = qa_pairs[idx]
        try:
            clear_memory() 
            result = chat_fn(qa["question"],retrieval)
            
            generated = result.response
            
            scores = judge_answer(
                question=qa["question"],
                reference_answer=qa["answer"],
                generated_answer=generated,
                content=qa["content"],
                llm=llm
            ).model_dump()
            
            time.sleep(12)  # Sleep for 12 seconds to avoid rate limiting

            scores["question"] = qa["question"]
            scores["source"] = result.source
            all_scores.append(scores)

        except Exception as e:
            logger.error(f"Error evaluating QA pair at index {idx} for video {video_id}: {e}")
            logger.error(e)
            
            dims=["correctness", "completeness", "faithfulness", "clarity"]
            if len(all_scores) > 0:
                avg_scores = {d: sum(s[d] for s in all_scores) / len(all_scores) for d in dims}
                avg_scores["overall"] = sum(avg_scores.values()) / len(dims)
            else:
                avg_scores={}
                
            checkpoint_data = {"scores": all_scores, "averages": avg_scores}
            save_json(checkpoint_data, checkpoint_path)
            
            raise e
            


    if not all_scores:
        return all_scores, {}

    # Aggregate
    dims = ["correctness", "completeness", "faithfulness", "clarity"]
    
    if len(all_scores)>0:
        avg_scores = {d: sum(s[d] for s in all_scores) / len(all_scores) for d in dims}
        avg_scores["overall"] = sum(avg_scores.values()) / len(dims)

    save_json({"scores": all_scores, "averages": avg_scores}, os.path.join(config.JUDGE_EVAL_RESULTS_DIR,f"{video_id}.json"))

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return all_scores, avg_scores
