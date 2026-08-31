from langsmith import traceable

import config
import os 
import numpy as np
from src.utils import save_json, deduplication
from evaluation.utils import compute_mrr, hit_rate_at_k, get_rank
from src.retrieval import rerank
from src.logger import get_logger


logger=get_logger(__name__)

def reranker_result_summary(results):
    pre_ranks=[r["pre_rank"] for r in results]
    post_ranks=[r["post_rank"] for r in results]
    dedup_ranks=[r["dedup_rank"] for r in results]

    valid_moves=[ r["movement"] for r in results if r["movement"] is not None]

    improved=sum( 1 for m in valid_moves if m>0)
    worsened=sum( 1 for m in valid_moves if m<0)
    unchanged=sum( 1 for m in valid_moves if m==0)

    avg_move=None
    if valid_moves:
        avg_move=sum(valid_moves) / len(valid_moves)


    dedup_moves = [r["dedup_movement"]for r in results if r["dedup_movement"] is not None]

    dedup_improved = sum(1 for m in dedup_moves if m > 0 )

    dedup_worsened = sum(1 for m in dedup_moves if m < 0)

    dedup_unchanged = sum(1 for m in dedup_moves if m == 0)

    dedup_avg_move = None

    if dedup_moves:
        dedup_avg_move = sum(dedup_moves) / len(dedup_moves)
    
    total_retrieval_docs = sum(r["retrieval_docs"] for r in results)
    total_dedup_docs = sum(r["dedup_docs"] for r in results)
    total_removed_docs = sum(r["removed_docs"] for r in results)

    removal_rate = (
        total_removed_docs / total_retrieval_docs
        if total_retrieval_docs
        else 0
    )

    summary = {
        "retrieval":{
            "mrr": compute_mrr(pre_ranks),
            "hit_rate":{
                k: hit_rate_at_k(pre_ranks, k) for k in [1, 3, 5]
            },
        "reranker":{
           "mrr": compute_mrr(post_ranks),
            "hit_rate":{
                k: hit_rate_at_k(post_ranks, k) for k in [1, 3, 5]
            },  
            "rank_movement": {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
            "average_movement": avg_move
                },
            },
        "reranker_with_dedup":{
            "mrr": compute_mrr(dedup_ranks),
            "hit_rate":{
                k: hit_rate_at_k(dedup_ranks, k) for k in [1, 3, 5]
            },
            "rank_movement": {
            "improved": dedup_improved,
            "worsened": dedup_worsened,
            "unchanged": dedup_unchanged,
            "average_movement": dedup_avg_move
                 },
            },
        "deduplication": {
            "total_retrieval_docs": total_retrieval_docs,
            "total_dedup_docs": total_dedup_docs,
            "total_removed_docs": total_removed_docs,
            "removal_rate": removal_rate
            }    
        }
    }

    
    
    return summary

@traceable(name="evaluate_reranker")
def evaluate_reranker(qa_pairs,retrieval,video_id):
  results=[]
  retriever=retrieval['ensemble_retriever']
  reranker=retrieval['reranker']
  embedder = retrieval['text_retriever'].vectorstore.embeddings

  for qa in qa_pairs:

      query=qa["question"]
      target_start=qa["start"]

      # Before reranking
      pre_docs=retriever.invoke(query)

      # similarities = get_pairwise_similarities(pre_docs, embedder)
      # all_similarities.extend(similarities) 

      pre_rank=get_rank(target_start,pre_docs)

      # Reranking without dedpulication 
      post_docs=rerank(query,pre_docs,reranker)
      post_rank=get_rank(target_start,post_docs)
    
      # Reranking with dedpulication 
      dedup_docs=deduplication(pre_docs,embedder)
      dedup_rerank_docs=rerank(query,dedup_docs,reranker)
      dedup_rank=get_rank(target_start,dedup_rerank_docs)


      movement=None

      if pre_rank is not None and post_rank is not None:
          movement=pre_rank-post_rank

      dedup_movement=None

      if pre_rank is not None and dedup_rank is not None:
        dedup_movement=pre_rank-dedup_rank 

      results.append({
          "question": query,
          "pre_rank": pre_rank,
          "post_rank": post_rank,
          "dedup_rank": dedup_rank,
          "movement": movement,
          "dedup_movement": dedup_movement,
          "retrieval_docs": len(pre_docs),
          "dedup_docs": len(dedup_docs),
          "removed_docs": len(pre_docs)-len(dedup_docs)
      })
  
  summary = reranker_result_summary(results)
  save_json(summary, os.path.join(config.RERANK_EVAL_RESULTS_DIR,f"{video_id}.json"))
  return summary