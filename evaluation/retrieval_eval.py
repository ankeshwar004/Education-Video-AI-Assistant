from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langsmith import traceable

import config
import os

from evaluation.utils import remove_duplicates,compute_mrr, hit_rate_at_k, get_rank
from evaluation.prompts import multi_query_prompt
from src.retrieval import rerank
from src.utils import save_json
from src.logger import get_logger


logger=get_logger(__name__)

@traceable(name="multiquery_retrieve")
def multiquery_retrieve(question,retriever,eval_llm):  

  query_chain=multi_query_prompt|eval_llm|StrOutputParser()

  response=query_chain.invoke({"question": question})

  queries=[question] + response.split("\n")

  all_docs=[]

  for q in queries:
      docs=retriever.invoke(q)
      all_docs.extend(docs)

  return all_docs


@traceable(name="evaluate_retrieval")
def evaluate_retrieval(qa_pairs, retrieval_fn, k_values=[1, 3, 5]):

    ranks=[]
    for qa in qa_pairs:
        query = qa["question"]
        source_start = qa["start"]
        
        docs = retrieval_fn(query)
        #With one groutnd truth cunk recall@k=hit@k

        # MRR
        rank=get_rank(source_start,docs)
        ranks.append(rank)

    hit_rates={k: hit_rate_at_k(ranks, k) for k in k_values}
    mrr=compute_mrr(ranks)

    return {"hit_rate": hit_rates, "mrr": mrr}


def evaluate_retrieval_configs(qa_pairs,retrieval,eval_llm,video_id):
    bm25_retriever=retrieval['bm25_retriever']
    text_retriever=retrieval['text_retriever']
    ensemble_retriever=retrieval['ensemble_retriever']
    reranker=retrieval['reranker']
    
    def bm25_only(query):
        return bm25_retriever.invoke(query)

    def vector_only(query):
        return text_retriever.invoke(query)

    def ensemble_only(query):
        return ensemble_retriever.invoke(query)

    def full_pipeline(query):
        docs = ensemble_retriever.invoke(query)
        return rerank(query,docs,reranker)

    def multiquery_full_pipeline(query):
        docs=multiquery_retrieve(query,ensemble_retriever,eval_llm)
        docs=remove_duplicates(docs)
        docs=rerank(query,docs,reranker)
        return docs

    configs = {
        "bm25_only":bm25_only,
        "vector_only":vector_only,
        "ensemble_only":ensemble_only,
        "ensemble_rerank":full_pipeline,
        "ensemble_multiquery_rerank":multiquery_full_pipeline
    }

    retrieval_results = {}
    for name, fn in configs.items():
        retrieval_results[name] = evaluate_retrieval(qa_pairs, fn, k_values=[1, 3, 5])

    save_json(retrieval_results, os.path.join(config.RETRIEVAL_EVAL_RESULTS_DIR,f"{video_id}.json"))
    return retrieval_results
