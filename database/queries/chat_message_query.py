from database.connection import pool


def create_chat_message(session_id: str, content: str, role: str):

    internal_session_id_find_query = """
        SELECT id
        FROM chat_sessions
        WHERE session_id = %s;
    """
    
    create_message_query = """
        INSERT INTO chat_messages (
            session_id,
            content,
            role
        )
        VALUES (%s, %s, %s)
        RETURNING *;
    """
    
    update_session_query = """
        UPDATE chat_sessions
        SET updated_at = CURRENT_TIMESTAMP
        WHERE session_id = %s;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(internal_session_id_find_query, (session_id,))
            session = cur.fetchone()
            
            if session is None:
                raise ValueError("Chat session not found")
            
            internal_session_id = session[0]
            cur.execute(create_message_query, (internal_session_id, content, role))

            message = cur.fetchone()
            if message is None:
                raise ValueError("Failed to create chat message")
            
            cur.execute(update_session_query, (session_id,))
            conn.commit()

    return message


def get_chat_messages(session_id: str):

    query = """
        SELECT *
        FROM chat_messages cm
        JOIN chat_sessions cs ON cm.session_id = cs.id
        WHERE cs.session_id = %s
        ORDER BY cm.created_at ASC;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            return cur.fetchall()


def delete_chat_messages(session_id: str):

    query = """
        DELETE FROM chat_messages
        WHERE session_id = (
            SELECT id
            FROM chat_sessions
            WHERE session_id = %s
        )
        RETURNING *;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            messages = cur.fetchall()
        conn.commit()

    return messages
