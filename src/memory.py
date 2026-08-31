import json
import config
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from src.llm import summarize_llm
from src.prompts import summarize_prompt
from src.cache import redis_client



def history_key(session_id):
    return f"memory:{session_id}:history"


def summary_key(session_id):
    return f"memory:{session_id}:summary"


def get_messages(session_id):
    messages=redis_client.lrange(history_key(session_id),0,-1)
    return [json.loads(message) for message in messages]


def add_message(session_id, role, content):
    message={"role": role,"content": content}

    redis_client.rpush(history_key(session_id),json.dumps(message))


def get_summary(session_id):
    return redis_client.get(summary_key(session_id)) or ""


def set_summary(session_id, summary):
    redis_client.set(summary_key(session_id),summary)


@traceable(name="Update Summary")
def update_summary(session_id, messages):

    old_summary=get_summary(session_id)

    chain=summarize_prompt|summarize_llm|StrOutputParser()

    formatted="\n".join(f"{msg['role']}:{msg['content']}"for msg in messages)

    new_summary=chain.invoke({
        "summary": old_summary,
        "new_messages": formatted
    })

    set_summary(session_id,new_summary)

    return new_summary


def update_memory(session_id,query,llm_response,message_window_size=config.MAX_TURNS):

    max_messages=message_window_size*2

    messages=get_messages(session_id)

    if len(messages) > max_messages:

        old_messages=messages[:-max_messages]
        update_summary(session_id,old_messages)

        redis_client.ltrim( history_key(session_id),-max_messages,-1)

    add_message(session_id,"user",query)

    add_message(session_id,"assistant",llm_response.response)


def clear_memory(session_id):

    redis_client.delete(history_key(session_id),summary_key(session_id))