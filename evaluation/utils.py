

def remove_duplicates(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        key = doc.page_content

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


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
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)