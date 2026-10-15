from database.connection import pool
from database.queries.chat_session_query import owner_filter


def get_recent_messages(session_id: str, limit: int, principal=None):
    
    if principal is not None:
        owner_clause, owner_params = owner_filter(principal)
        query = f"""
            SELECT cm.role, cm.content
            FROM chat_messages cm
            JOIN chat_sessions cs ON cm.session_id = cs.session_id
            WHERE cm.session_id = %s AND {owner_clause}
            ORDER BY cm.created_at DESC, cm.id DESC
            LIMIT %s;
        """
        params = (session_id, *owner_params, limit)
    else:
        query = """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s;
        """
        params = (session_id, limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def get_all_messages(session_id: str, principal=None):
    
    if principal is not None:
        owner_clause, owner_params = owner_filter(principal)
        query = f"""
            SELECT cm.id, cm.role, cm.content, cm.created_at
            FROM chat_messages cm
            JOIN chat_sessions cs ON cm.session_id = cs.session_id
            WHERE cm.session_id = %s AND {owner_clause}
            ORDER BY cm.created_at ASC, cm.id ASC;
        """
        params = (session_id, *owner_params)
    else:
        query = """
            SELECT id, role, content, created_at
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC, id ASC;
        """
        params = (session_id,)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()


def get_session_summary(session_id: str, principal=None):
    
    if principal is not None:
        owner_clause, owner_params = owner_filter(principal)
        query = f"""
            SELECT summary, summary_through_message_id
            FROM chat_sessions
            WHERE session_id = %s AND {owner_clause};
        """
        params = (session_id, *owner_params)
    else:
        query = """
            SELECT summary, summary_through_message_id
            FROM chat_sessions
            WHERE session_id = %s;
        """
        params = (session_id,)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
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
    principal=None,
):
    if principal is not None:
        owner_clause, owner_params = owner_filter(principal)
        query = f"""
            UPDATE chat_sessions
            SET summary = %s,
                summary_through_message_id = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s AND {owner_clause};
        """
        params = (summary, summary_through_message_id, session_id, *owner_params)
    else:
        query = """
            UPDATE chat_sessions
            SET summary = %s,
                summary_through_message_id = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s;
        """
        params = (summary, summary_through_message_id, session_id)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()


def clear_session_memory(session_id: str, principal=None):
    
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

    if principal is not None:
        owner_clause, owner_params = owner_filter(principal)
        # Only delete messages and clear summary if the session belongs to the owner
        delete_messages_query = f"""
            DELETE FROM chat_messages
            WHERE session_id = %s
              AND session_id IN (
                  SELECT session_id FROM chat_sessions
                  WHERE session_id = %s AND {owner_clause}
              );
        """
        clear_summary_query = f"""
            UPDATE chat_sessions
            SET summary = NULL,
                summary_through_message_id = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s AND {owner_clause};
        """
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_messages_query, (session_id, session_id, *owner_params))
                cur.execute(clear_summary_query, (session_id, *owner_params))
            conn.commit()
    else:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_messages_query, (session_id,))
                cur.execute(clear_summary_query, (session_id,))
            conn.commit()
