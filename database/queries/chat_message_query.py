from database.connection import pool


def create_chat_message(session_id: str, content: str, role: str):

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
            cur.execute(create_message_query, (session_id, content, role))

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
        JOIN chat_sessions cs ON cm.session_id = cs.session_id
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
        WHERE session_id = %s
        RETURNING *;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            messages = cur.fetchall()
        conn.commit()

    return messages
