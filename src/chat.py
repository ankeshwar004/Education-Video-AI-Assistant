import time
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableBranch,RunnablePassthrough
from langsmith import traceable


import config
from src import retrieval
from src.logger import get_logger
from src.prompts import contextualize_qa_prompt, qa_prompt
from src.memory import chat_history, update_memory
from src.llm import build_multimodal_message,decision_strutured_llm, main_llm, main_strutured_llm
from src.utils import load_images
from src.prompts import contextualize_qa_prompt, decision_prompt, qa_prompt



logger = get_logger(__name__)



@traceable(name="Query Needs Images")
def query_needs_images(query):
  decision_chain=decision_prompt|decision_strutured_llm
  decision=decision_chain.with_config({"run_name":"Decision LLM"}).invoke({"query":query})
  return decision.answer


#Option 1

@traceable(name="Python chat pipeline")
def chat(query):

  if chat_history.messages:
    contextualize_chain=contextualize_qa_prompt|main_llm|StrOutputParser()
    standalone_ques=contextualize_chain.with_config({"run_name":"Contextualize Query"}).invoke({
        "query":query,
        "chat_history":chat_history.messages
    })
  else:
    standalone_ques=query
    
  logger.info("StandAlone Ques:\n%s",standalone_ques)

  docs=retrieval.ensemble_retriever.invoke(standalone_ques)

  docs=retrieval.rerank(standalone_ques,docs,k=config.CHAT_RERANK_K)

  text_context=retrieval.build_text_context(docs)

  chunk_ids=[doc.metadata['chunk_id'] for doc in docs]

  frames=retrieval.frame_retriever(standalone_ques,chunk_ids)
 
  frames_metadata=frames['metadatas'][0]

  ocr_context=retrieval.build_ocr_context(frames_metadata)
  logger.info("Length of OCR Contex: %s",len(ocr_context))

  needs_vision = query_needs_images(standalone_ques)
  logger.info("Needs Vision?: %s",needs_vision)

  images=[]
  if needs_vision:
    images=load_images(frames_metadata)
    logger.info("Length of load images %s",len(images))


  prompt_value=qa_prompt.invoke({
      "context":text_context,
      "ocr_context":ocr_context,
      "query":standalone_ques,
      "chat_history":chat_history.messages
  })
  prompt_value_messages=prompt_value.messages

  messages=build_multimodal_message(prompt_value_messages,images)
  logger.info("Length of messages: %s",len(messages))

  try:
    logger.info("Calling LLM")
    start = time.time()
    llm_response=main_strutured_llm.invoke(messages)
    end = time.time()

    logger.info("Time taken for LLM response: %s",end-start)
  except Exception as e:
    logger.error(type(e))
    logger.error(e)
    raise

  update_memory(standalone_ques,llm_response)

  return llm_response


#Option 2

def lcel_chat(query):
    
    contextualize_chain=contextualize_qa_prompt|main_llm|StrOutputParser()
    standalone_query=RunnableBranch(
        (lambda x:bool(chat_history.messages),RunnablePassthrough.assign(
            query=RunnableLambda(
            lambda x: {"query":x["query"], "chat_history":chat_history.messages}
            )|contextualize_chain.with_config({"run_name":"Contextualize Query"})
            )),
        RunnablePassthrough()
        ).with_config({"run_name":"Standalone Query"})

    retriever_step=RunnablePassthrough.assign(
        docs=RunnableLambda(lambda x: x["query"])|retrieval.ensemble_retriever
        ).with_config({"run_name":"Ensemble Retriever"})

    reranker_step=RunnableLambda(
        lambda x: {**x, "docs":retrieval.rerank(x["query"],x["docs"],k=3)}
        ).with_config({"run_name":"Reranker"})

    retrieval_pipeline=retriever_step|reranker_step

    text_context_builder=RunnableLambda(
        lambda x: {**x,"text_context":retrieval.build_text_context(x["docs"]),
        "chunk_ids":[doc.metadata['chunk_id'] for doc in x["docs"]]}
        ).with_config({"run_name":"Build Text Context"})

    frame_retrival=RunnableLambda(
        lambda x: {**x,"frames":retrieval.frame_retriever(x["query"],x["chunk_ids"])}
        ).with_config({"run_name":"Frame Retriever"})

    ocr_context_builder=RunnableLambda(
        lambda x: {**x,"ocr_context":retrieval.build_ocr_context(x["frames"]["metadatas"][0])}
        ).with_config({"run_name":"Build OCR Context"})

    vision_decision=RunnablePassthrough.assign(
        need_vision=RunnableLambda(lambda x: query_needs_images(x["query"])) # inside it llm is called (i.e llm.invoke(query))
        ).with_config({"run_name":"Vision Decision"})

    vision_loader=RunnableBranch(
        (lambda x: x["need_vision"],RunnableLambda(
            lambda x: {**x,"images":load_images(x["frames"]["metadatas"][0])}
            ).with_config({"run_name":"Load Images"})),
        RunnableLambda(
            lambda x: {**x,"images":[]}
            ).with_config({"run_name":"No Images"})
        ).with_config({"run_name":"Vision Loader"})

    vision_pipeline=frame_retrival|ocr_context_builder|vision_decision|vision_loader

    prompt_builder=RunnablePassthrough.assign(
        prompt_value_messages=RunnableLambda(
            lambda x: {"context":x["text_context"],
                    "ocr_context":x["ocr_context"],
                    "query":x["query"],
                    "chat_history":chat_history.messages}
        )|qa_prompt).with_config({"run_name":"Prompt Builder"})

    message_builder=RunnableLambda(
        lambda x: {**x,"messages":build_multimodal_message(x["prompt_value_messages"],x["images"])}
        ).with_config({"run_name":"Build Message"}) 

    llm_result=(
        RunnableLambda(lambda x: x['messages']) | main_strutured_llm
        ).with_config({"run_name":"LLM Result"})


    chat_pipeline=(standalone_query|retrieval_pipeline|text_context_builder|
    vision_pipeline|prompt_builder|message_builder|llm_result).with_config({"run_name":"Chat Pipeline"})

    response=chat_pipeline.invoke({"query":query})
    update_memory(query,response) 

    return response
