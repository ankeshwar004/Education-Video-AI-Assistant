import random

from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

import config


class QAPairs(BaseModel):
  question: str
  answer: str


class QAPairsList(BaseModel):
  pairs: list[QAPairs]


def create_eval_llm():
    return ChatOpenRouter(model=config.EVAL_LLM_MODEL)


def generate_qa_pairs(docs,n=2,eval_llm=None):
  qa_pairs=[]
  if eval_llm is None:
    eval_llm=create_eval_llm()

  generation_prompt = PromptTemplate.from_template(
      """You are creating an evaluation dataset for an educational video assistant.

  Given this transcript segment from a lecture video:
  {content}

  Generate {n} question-answer pairs where:
  - The question is something a student would genuinely ask
  - The answer must be completely answerable using only the information contained in this segment.
  - Do not generate questions that require information from previous or subsequent transcript segments.
  - Questions should test understanding, not verbatim recall
  - Mix types: conceptual ("what is X"), factual ("what did the speaker say about X"), and explanatory ("how does X work")

  Return ONLY a JSON array, no other text:
  [{{"question": "...", "answer": "..."}}]"""
  )

  structured_eval_llm=eval_llm.with_structured_output(QAPairsList)
  for doc in docs:
    chain=generation_prompt|structured_eval_llm

    response=chain.invoke({"content":doc.page_content,"n":n})

    #Assuming The chunk that generated the question is the ground-truth chunk.(Not perfect but simplify)

    for pair in response.pairs:
      qa_pairs.append({
      "question": pair.question,
      "answer": pair.answer,
      "start": doc.metadata["start"],
      "end": doc.metadata["end"],
      "content": doc.page_content
      })
  return qa_pairs


def sample_and_generate(text_docs, n=2):
    eval_docs=random.sample(text_docs, min(50,len(text_docs)//2))
    qa_pairs=generate_qa_pairs(eval_docs,n=n)
    print(f"Length:{len(qa_pairs)} from docs of length {len(eval_docs)}")
    return qa_pairs
