from langsmith import traceable

import config
import os 
from src.utils import save_json
from evaluation.utils import compute_mrr, hit_rate_at_k, get_rank
from src.logger import get_logger


logger=get_logger(__name__)

def reranker_result_summary(results):
    pre_ranks=[r["pre_rank"] for r in results]
    post_ranks=[r["post_rank"] for r in results]

    valid_moves=[ r["movement"] for r in results if r["movement"] is not None]

    improved=sum( 1 for m in valid_moves if m>0)
    worsened=sum( 1 for m in valid_moves if m<0)
    unchanged=sum( 1 for m in valid_moves if m==0)

    avg_move=None
    if valid_moves:
        avg_move=sum(valid_moves) / len(valid_moves)

    summary = {
        "mrr_before": compute_mrr(pre_ranks),
        "mrr_after": compute_mrr(post_ranks),
        "hit_rate_before": {
            k: hit_rate_at_k(pre_ranks, k) for k in [1, 3, 5]
        },
        "hit_rate_after": {
            k: hit_rate_at_k(post_ranks, k) for k in [1, 3, 5]
        },
        "rank_movement": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
            "average_movement": avg_move
        },
        "per_question": results
    }
    
    
    return summary

@traceable(name="evaluate_reranker")
def evaluate_reranker(qa_pairs,retrieval,video_id):
  results=[]
  retriever=retrieval['ensemble_retriever']
  reranker=retrieval['reranker']

  for qa in qa_pairs:

      query=qa["question"]
      target_start=qa["start"]

      # Before reranking
      pre_docs=retriever.invoke(query)
      pre_rank=get_rank(target_start,pre_docs)

      # After reranking
      pairs=[(query, doc.page_content) for doc in pre_docs]
      scores=reranker.predict(pairs)
      reranked=sorted(zip(pre_docs,scores),key=lambda x:x[1],reverse=True)

      post_docs=[doc for doc, score in reranked]
      post_rank=get_rank(target_start,post_docs)

      movement=None

      if pre_rank is not None and post_rank is not None:
          movement=pre_rank-post_rank

      results.append({
          "question": query,
          "pre_rank": pre_rank,
          "post_rank": post_rank,
          "movement": movement
      })
  
  
  summary = reranker_result_summary(results)
  save_json(summary, os.path.join(config.RERANK_EVAL_RESULTS_DIR,f"{video_id}.json"))
  return summary