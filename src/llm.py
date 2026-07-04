from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage

from src.base_models import Answer,Decision

import config


main_llm=ChatGoogleGenerativeAI(model=config.MAIN_LLM_MODEL)

main_strutured_llm=main_llm.with_structured_output(Answer)

summarize_llm=ChatOpenRouter(model=config.SUMMARY_LLM_MODEL)

decision_llm=ChatGroq(model=config.DECISION_LLM_MODEL,temperature=0)
decision_strutured_llm=decision_llm.with_structured_output(Decision)

eval_llm=ChatOpenRouter(model=config.EVAL_LLM_MODEL)

def build_multimodal_message(prompt_value_messages,images,provider="gemini"):
  messages=prompt_value_messages

  if len(images)>0:
        content = [{
                "type": "text",
                "text": messages[-1].content,
            }]

        for img in images:
          if provider=="openrouter":
            content.append({
                "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
            })

          elif provider=="gemini":
            content.append({
                "type":"image",
                "source_type":"base64",
                "mime_type":"image/jpeg",
                "data":img
            })

        messages[-1]=HumanMessage(content=content)

  return messages
