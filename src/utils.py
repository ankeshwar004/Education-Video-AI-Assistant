import base64
from difflib import SequenceMatcher
import json
import os
import tiktoken
import config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def format_time(seconds):
  hours=int(seconds//(60*60))
  minutes=int((seconds%(60*60))//60)
  seconds=int(seconds%60)
  return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours>0 else f"{minutes:02d}:{seconds:02d}"


def to_seconds(ts):
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = [float(p) for p in str(ts).split(':')]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds
  
  
  

def is_similar(text,seen_texts,threshold=0.85):
  for seen_text in seen_texts:
    ratio=SequenceMatcher(None,text,seen_text).ratio()
    if ratio>threshold:
      return True
  return False



enc=tiktoken.get_encoding("cl100k_base")


def token_count(text):
  return len(enc.encode(text))
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_images(frames_metadata):
  images=[]
  for frame in frames_metadata:
    path=frame['frame_path']
    with open(path,"rb") as f:
            images.append(base64.b64encode(f.read()).decode())
  return images



def deduplication(docs, embedder, threshold=0.90):

    unique_docs = []
    unique_embeddings = []

    embeddings = embedder.embed_documents(
        [doc.page_content for doc in docs]
    )

    for doc, embedding in zip(docs, embeddings):

        if not unique_embeddings:
            unique_docs.append(doc)
            unique_embeddings.append(embedding)
            continue

        similarities = cosine_similarity(
            [embedding],
            unique_embeddings
        )[0]

        if max(similarities) < threshold:
            unique_docs.append(doc)
            unique_embeddings.append(embedding)

    return unique_docs

