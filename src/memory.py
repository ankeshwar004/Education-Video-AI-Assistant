import config
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from src.llm import summarize_llm
from src.prompts import summarize_prompt
from database.queries.chat_message_query import create_chat_message
from database.queries.chat_session_query import create_chat_session, get_chat_session
from database.queries.memory_query import (
    clear_session_memory,
    get_all_messages as query_all_messages,
    get_recent_messages,
    get_session_summary,
    update_session_summary,
)
from services.auth_service import new_anonymous_principal


def initialize_memory(session_id, video_id, title=None, principal=None):
    principal = principal or new_anonymous_principal()

    session = get_chat_session(session_id, principal)
    if session is not None:
        if session["video_id"] != video_id:
            raise ValueError("Session belongs to a different video")
        return session

    return create_chat_session(session_id, video_id, principal, title)


def get_messages(session_id):

    max_messages = config.MAX_TURNS * 2
    summary = get_summary(session_id)

    rows = get_recent_messages(session_id, max_messages)

    messages = [
        {"role": row["role"], "content": row["content"]}
        for row in reversed(rows)
    ]

    if summary:
        messages.insert(0, {
            "role": "system",
            "content": f"Conversation summary:\n{summary}"
        })

    return messages


def get_all_messages(session_id):
    rows = query_all_messages(session_id)

    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]


def add_message(session_id, role, content):
    create_chat_message(session_id, content, role)


def get_summary(session_id):
    return get_session_summary(session_id)["summary"] or ""


def set_summary(session_id, summary, summary_through_message_id):
    update_session_summary(session_id,summary,summary_through_message_id)


@traceable(name="Update Summary")
def update_summary(session_id, messages, summary_through_message_id):

    old_summary = get_summary(session_id)

    chain = summarize_prompt | summarize_llm | StrOutputParser()

    formatted = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

    new_summary = chain.invoke({"summary": old_summary,"new_messages": formatted})

    set_summary(session_id, new_summary, summary_through_message_id)

    return new_summary


def update_memory( session_id, query, llm_response, message_window_size=config.MAX_TURNS):

    add_message(session_id,"user",query)

    add_message(session_id,"assistant",llm_response.response)

    all_messages = get_all_messages(session_id)

    max_messages = message_window_size * 2

    if len(all_messages) > max_messages:
        summary_state = get_session_summary(session_id)
        marker = summary_state["summary_through_message_id"]
        old_messages = [
            row for row in all_messages[:-max_messages]
            if marker is None or row["id"] > marker
        ]

        if old_messages:
            update_summary(
                session_id,
                [
                    {"role": row["role"], "content": row["content"]}
                    for row in old_messages
                ],
                old_messages[-1]["id"],
            )


def clear_memory(session_id):

    clear_session_memory(session_id)
