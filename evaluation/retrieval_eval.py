from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import config
from src.retrieval import rerank


def multiquery_retrieve(question,retriever,eval_llm):
  query_prompt=PromptTemplate(
    template="""
    You are helping retrieve video transcript chunks.

    Generate 2 alternative queries that may use
    different terminology but ask the same thing.

    Question:
    {question}

    Return one query per line.
    """,
        input_variables=["question"]
  )

  query_chain=query_prompt|eval_llm|StrOutputParser()

  response=query_chain.invoke({"question": question})

  queries=[question] + response.split("\n")

  all_docs=[]

  for q in queries:
      docs=retriever.invoke(q)
      all_docs.extend(docs)

  return all_docs


def remove_duplicates(docs):

    seen = set()
    unique_docs = []

    for doc in docs:
        key = doc.page_content

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def evaluate_retrieval(qa_pairs, retrieval_fn, k_values=[1, 3, 5]):

    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for qa in qa_pairs:
        query = qa["question"]
        source_start = qa["start"]

        retrieved = retrieval_fn(query)
        retrieved_starts = [doc.metadata["start"] for doc in retrieved]

        #With one groutnd truth cunk recall@k=hit@k

        # Hit Rate@k
        for k in k_values:
            if source_start in retrieved_starts[:k]:
                hit_counts[k] += 1

        # MRR
        if source_start in retrieved_starts:
            rank = retrieved_starts.index(source_start) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(qa_pairs)
    hit_rates = {k: hit_counts[k] / n for k in k_values}
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return {"hit_rate": hit_rates, "mrr": mrr}


def evaluate_configs(qa_pairs, retrieval_components, eval_llm):
    bm25_retriever=retrieval_components["bm25_retriever"]
    text_retriever=retrieval_components["text_retriever"]
    ensemble_retriever=retrieval_components["ensemble_retriever"]
    reranker=retrieval_components["reranker"]

    def bm25_only(query):
        return bm25_retriever.invoke(query)

    def vector_only(query):
        return text_retriever.invoke(query)

    def ensemble_only(query):
        return ensemble_retriever.invoke(query)

    def full_pipeline(query):
        docs = ensemble_retriever.invoke(query)
        return rerank(query,docs,reranker,k=config.RERANK_K)

    def multiquery_full_pipeline(query):
        docs=multiquery_retrieve(query,ensemble_retriever,eval_llm)
        docs=remove_duplicates(docs)
        docs=rerank(query,docs,reranker,k=config.RERANK_K)
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
        result = evaluate_retrieval(qa_pairs, fn, k_values=[1, 3, 5])
        retrieval_results[name] = result
        print(f"\n{name}")
        print(f"  MRR:          {result['mrr']:.3f}")
        print(f"  Hit Rate@1:   {result['hit_rate'][1]:.3f}")
        print(f"  Hit Rate@3:   {result['hit_rate'][3]:.3f}")
        print(f"  Hit Rate@5:   {result['hit_rate'][5]:.3f}")
    return retrieval_results


def get_rank(target_start,docs):
  start=[doc.metadata['start'] for doc in docs]
  return start.index(target_start)+1 if target_start in start else None


def compute_mrr(ranks):
    scores = []

    for rank in ranks:
        if rank is None:
            scores.append(0.0)
        else:
            scores.append(1.0 / rank)

    return sum(scores) / len(scores)


def hit_rate_at_k(ranks, k):
    hits = sum(
        1 for rank in ranks
        if rank is not None and rank <= k
    )

    return hits / len(ranks)


def evaluate_reranker(qa_pairs,retriever,reranker):
  results=[]

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

  pre_ranks=[r["pre_rank"] for r in results]
  post_ranks=[r["post_rank"] for r in results]

  print("\n===== Retrieval vs Reranker =====")

  print(f"MRR Before: {compute_mrr(pre_ranks):.3f}")
  print(f"MRR After : {compute_mrr(post_ranks):.3f}")

  for k in [1, 3, 5]:
      print(f"Hit@{k} Before: "f"{hit_rate_at_k(pre_ranks,k):.3f}")

      print(f"Hit@{k} After : "f"{hit_rate_at_k(post_ranks,k):.3f}")

  valid_moves=[ r["movement"] for r in results if r["movement"] is not None]

  improved=sum( 1 for m in valid_moves if m > 0)

  worsened=sum( 1 for m in valid_moves if m < 0)

  unchanged=sum( 1 for m in valid_moves if m == 0)

  print("\nRank Movement")

  print(f"Improved : {improved}")
  print(f"Worsened : {worsened}")
  print(f"Unchanged: {unchanged}")

  if valid_moves:
      avg_move=sum(valid_moves) / len(valid_moves)

      print(f"Average movement: "f"{avg_move:+.2f}")

  return results
