from pydantic import BaseModel
from langsmith import traceable

import config
import os
from src.utils import save_json
from evaluation.prompts import judge_prompt
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
def evaluate_llm_as_judge(qa_pairs,chat_fn,retrieval,llm,video_id):

    all_scores = []
    failures = []

    for qa in qa_pairs:
        try:
            result = chat_fn(qa["question"],retrieval)
            generated = result.response

            scores = judge_answer(
                question=qa["question"],
                reference_answer=qa["answer"],
                generated_answer=generated,
                content=qa["content"],
                llm=llm
            )

            scores["question"] = qa["question"]
            scores["source"] = result.source
            all_scores.append(scores)

        except Exception as e:
            failures.append({"question": qa["question"], "error": str(e)})
            print(f"Error evaluating question: {qa['question']}")
            print(e)


    if not all_scores:
        return all_scores, {}, failures

    # Aggregate
    dims = ["correctness", "completeness", "faithfulness", "clarity"]
    
    if len(all_scores)>0:
        avg_scores = {d: sum(s[d] for s in all_scores) / len(all_scores) for d in dims}
        avg_scores["overall"] = sum(avg_scores.values()) / len(dims)


    save_json({"scores": all_scores, "averages": avg_scores, "failures": failures}, os.path.join(config.JUDGE_EVAL_RESULTS_DIR,f"{video_id}.json"))

    return all_scores, avg_scores, failures
