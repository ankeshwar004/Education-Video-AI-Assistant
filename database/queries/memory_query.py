from database.connection import pool


def get_recent_messages(session_id: str, limit: int):
    
    query = """
        SELECT role, content
        FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at DESC, id DESC
        LIMIT %s;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id, limit))
            return cur.fetchall()


def get_all_messages(session_id: str):
    
    query = """
        SELECT id, role, content, created_at
        FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at ASC, id ASC;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            return cur.fetchall()


def get_session_summary(session_id: str):
    
    query = """
        SELECT summary, summary_through_message_id
        FROM chat_sessions
        WHERE session_id = %s;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            row = cur.fetchone()

    if row is None:
        return {"summary": "", "summary_through_message_id": None}

    return {
        "summary": row["summary"] or "",
        "summary_through_message_id": row["summary_through_message_id"],
    }


def update_session_summary(
    session_id: str,
    summary: str,
    summary_through_message_id: int,
):
    query = """
        UPDATE chat_sessions
        SET summary = %s,
            summary_through_message_id = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE session_id = %s;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query,(summary, summary_through_message_id, session_id),)
        conn.commit()


def clear_session_memory(session_id: str):
    
    delete_messages_query = """
        DELETE FROM chat_messages
        WHERE session_id = %s;
    """
    clear_summary_query = """
        UPDATE chat_sessions
        SET summary = NULL,
            summary_through_message_id = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE session_id = %s;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(delete_messages_query, (session_id,))
            cur.execute(clear_summary_query, (session_id,))
        conn.commit()
