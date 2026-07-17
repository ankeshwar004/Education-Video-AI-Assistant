from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from src.llm import summarize_llm
from src.prompts import summarize_prompt
import config

chat_history=InMemoryChatMessageHistory()
chat_summary=""
    

@traceable(name="Update Summary")
def update_summary(messages):
  global chat_summary

  chain=summarize_prompt|summarize_llm|StrOutputParser()

  formatted = "\n".join( f"{msg.type}: {msg.content}" for msg in messages)

  chat_summary=chain.invoke({
      "summary":chat_summary,
      "new_messages":formatted
  })
  return chat_summary


def update_memory(query,llm_response,message_window_size=config.MAX_TURNS):
  MAX_TURNS=message_window_size

  if len(chat_history.messages)>MAX_TURNS*2:
    old_messages=chat_history.messages[:-(MAX_TURNS*2)]
    update_summary(old_messages)
    del chat_history.messages[:-(MAX_TURNS*2)]

  chat_history.add_user_message(query)
  chat_history.add_ai_message(llm_response.response)

  return

def clear_memory():
  global chat_summary
  chat_history.clear()
  chat_summary=""
  return
