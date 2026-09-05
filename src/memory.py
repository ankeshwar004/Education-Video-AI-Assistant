import config
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from src.llm import summarize_llm
from src.prompts import summarize_prompt
from database.connection import pool


def get_messages(session_id):
    """
    Get the recent messages for active chat context.

    PostgreSQL keeps ALL messages permanently.
    Only the latest MAX_TURNS*2 messages are returned here
    for the LLM conversation context.
    """

    max_messages = config.MAX_TURNS * 2

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, max_messages)
            )

            rows = cur.fetchall()

    # We queried newest → oldest, so reverse them
    # to return chronological order.
    return [
        {"role": role, "content": content}
        for role, content in reversed(rows)
    ]


def get_all_messages(session_id):
    """
    Get the complete chat history of a session.

    Useful when the user opens an old session.
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,)
            )

            rows = cur.fetchall()

    return [
        {
            "role": role,
            "content": content,
            "created_at": created_at
        }
        for role, content, created_at in rows
    ]


def add_message(session_id, role, content):
    """
    Permanently store a message in PostgreSQL.
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages
                    (session_id, role, content)
                VALUES
                    (%s, %s, %s)
                """,
                (session_id, role, content)
            )

            conn.commit()


def get_summary(session_id):
    """
    Get the current summary of a chat session.
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary
                FROM chat_sessions
                WHERE session_id = %s
                """,
                (session_id,)
            )

            row = cur.fetchone()

    if row is None:
        return ""

    return row[0] or ""


def set_summary(session_id, summary):
    """
    Save/update the summary of a chat session.
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chat_sessions
                SET summary = %s,
                    updated_at = NOW()
                WHERE session_id = %s
                """,
                (summary, session_id)
            )

            conn.commit()


@traceable(name="Update Summary")
def update_summary(session_id, messages):
    """
    Summarize older messages and store the summary
    in PostgreSQL.
    """

    old_summary = get_summary(session_id)

    chain = summarize_prompt | summarize_llm | StrOutputParser()

    formatted = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in messages
    )

    new_summary = chain.invoke(
        {
            "summary": old_summary,
            "new_messages": formatted
        }
    )

    set_summary(session_id, new_summary)

    return new_summary


def update_memory(
    session_id,
    query,
    llm_response,
    message_window_size=config.MAX_TURNS
):
    """
    Save the user query and assistant response.

    PostgreSQL stores the complete history.

    The old Redis implementation removed old messages after
    MAX_TURNS. We DO NOT do that anymore because PostgreSQL
    is our permanent history store.
    """

    # Store user message
    add_message(
        session_id,
        "user",
        query
    )

    # Store assistant response
    add_message(
        session_id,
        "assistant",
        llm_response.response
    )

    # Get all messages to determine whether summarization
    # is required.
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,)
            )

            rows = cur.fetchall()

    messages = [
        {"role": role, "content": content}
        for role, content in rows
    ]

    max_messages = message_window_size * 2

    if len(messages) > max_messages:
        old_messages = messages[:-max_messages]

        update_summary(
            session_id,
            old_messages
        )


def clear_memory(session_id):
    """
    Delete the complete chat session history.
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                DELETE FROM chat_messages
                WHERE session_id = %s
                """,
                (session_id,)
            )

            cur.execute(
                """
                UPDATE chat_sessions
                SET summary = NULL,
                    updated_at = NOW()
                WHERE session_id = %s
                """,
                (session_id,)
            )

            conn.commit()
