import base64
from difflib import SequenceMatcher
import json
import os
import config
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)


def make_dir(path):
    os.makedirs(path, exist_ok=True)

def format_time(seconds):
  hours=int(seconds//(60*60))
  minutes=int((seconds%(60*60))//60)
  seconds=int(seconds%60)
  return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours>0 else f"{minutes:02d}:{seconds:02d}"

def is_similar(text,seen_texts,threshold=0.85):
  for seen_text in seen_texts:
    ratio=SequenceMatcher(None,text,seen_text).ratio()
    if ratio>threshold:
      return True
  return False

def token_count(text,tokenizer):
    return len(tokenizer.encode(text))

def load_json(path):
    """Load JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """Save JSON to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_images(frames_metadata):
  images=[]
  for frame in frames_metadata:
    path=frame['frame_path']
    with open(path,"rb") as f:
            images.append(base64.b64encode(f.read()).decode())
  return images
