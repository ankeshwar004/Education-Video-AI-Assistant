import time
from typing import Literal

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field

import config
from src.logger import get_logger
from src.prompts import contextualize_qa_prompt, qa_prompt
from src.retrieval import build_ocr_context, build_text_context, rerank
from src.utils import load_images


logger = get_logger(__name__)


class Answer(BaseModel):
    response: str=Field(
        """
        A clear educational explanation written in the model's own words.
        Do not quote transcript verbatim unless necessary"""
        )

    timestamps: list[str]=Field(
        default_factory=list,
        description="Relevant timestamp ranges from the video. Empty list if answered from general knowledge only."
    )

    source: Literal["video","general_knowledge","hybrid"]=Field(
        description=(
            "'video'=answered from transcript context. "
            "'general_knowledge'=video didn't cover it, used prior knowledge. "
            "'hybrid'=used both."
        )
    )

    key_takeway: list[str]=Field(
        description="Point out the important point only"
    )


class Decision(BaseModel):
  answer: bool


llm = None
strutured_llm = None
decision_llm = None
decision_strutured_llm = None
contextualize_chain = None
chat_history = InMemoryChatMessageHistory()
ensemble_retriever = None
reranker_model = None
clip_model = None
frame_db = None


def initialize_chat_pipeline(retrieval_components, clip, frame_collection):
    global llm
    global strutured_llm
    global decision_llm
    global decision_strutured_llm
    global contextualize_chain
    global ensemble_retriever
    global reranker_model
    global clip_model
    global frame_db

    llm=ChatOpenRouter(model=config.LLM_MODEL)
    strutured_llm=llm.with_structured_output(Answer)
    contextualize_chain=contextualize_qa_prompt|llm|StrOutputParser()

    decision_llm = ChatGoogleGenerativeAI(model=config.DECISION_LLM_MODEL)
    decision_strutured_llm=decision_llm.with_structured_output(Decision)

    ensemble_retriever=retrieval_components["ensemble_retriever"]
    reranker_model=retrieval_components["reranker"]
    clip_model=clip
    frame_db=frame_collection
    return chat


def build_multimodal_message(text_context,ocr_context,standalone_ques,images):
  prompt_value=qa_prompt.invoke({
      "context":text_context,
      "ocr_context":ocr_context,
      "query":standalone_ques,
      "chat_history":chat_history.messages
  })

  messages=prompt_value.messages

  if len(images)>0:
        content = [{
                "type": "text",
                "text": messages[-1].content,
            }]

        for img in images:
            content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                })

        messages[-1]=HumanMessage(content=content)

  return messages


def query_needs_images(query):
  decision_prompt = ChatPromptTemplate.from_template("""
  Determine whether the following question requires video frames.
  Question:
  {query}
  """)
  decision_chain=decision_prompt|decision_strutured_llm
  decision=decision_chain.invoke({"query":query})
  return decision.answer


def chat(query):
  """Answer a question using the transcript, OCR, frames, and chat history."""
  if any(value is None for value in [strutured_llm, contextualize_chain, ensemble_retriever, reranker_model, clip_model, frame_db]):
      raise RuntimeError("Chat pipeline is not initialized. Call initialize_chat_pipeline first.")

  if chat_history.messages:
    standalone_ques=contextualize_chain.invoke({
        "query":query,
        "chat_history":chat_history.messages
    })
  else:
    standalone_ques=query
  logger.info("StandAlone Ques:\n%s",standalone_ques)

  docs=ensemble_retriever.invoke(standalone_ques)

  docs=rerank(standalone_ques,docs,reranker_model,k=config.CHAT_RERANK_K)

  text_context=build_text_context(docs)

  chunk_ids=[doc.metadata['chunk_id'] for doc in docs]

  query_embedding=clip_model.encode(standalone_ques,convert_to_numpy=True,normalize_embeddings=True)

  frames=frame_db.query(
      query_embeddings=[query_embedding.tolist()],
      where={"chunk_id": {"$in": chunk_ids}},
      n_results=config.FRAME_RESULTS_K,
      include=["metadatas", "distances"]
  )

  frames_metadata=frames['metadatas'][0]

  ocr_context=build_ocr_context(frames_metadata)
  logger.info("Length of OCR Contex: %s",len(ocr_context))

  needs_vision = query_needs_images(standalone_ques)
  logger.info("Needs Vision?: %s",needs_vision)

  images=[]
  if needs_vision:
    images=load_images(frames_metadata)
    logger.info("Length of load images %s",len(images))

  messages=build_multimodal_message(text_context,ocr_context,standalone_ques,images)
  logger.info("Length of messages: %s",len(messages))

  try:
    logger.info("Calling LLM")
    start = time.time()
    result=strutured_llm.invoke(messages)
    end = time.time()

    logger.info("Time taken for LLM response: %s",end-start)
  except Exception as e:
    logger.error(type(e))
    logger.error(e)
    raise

  if len(chat_history.messages)>config.MAX_TURNS*2:
      del chat_history.messages[:-(config.MAX_TURNS*2)]

  chat_history.add_user_message(query)
  chat_history.add_ai_message(result.response)

  return result
