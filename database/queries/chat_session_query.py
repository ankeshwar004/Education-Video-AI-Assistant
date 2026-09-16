from database.connection import pool


def owner_filter(principal):
    if principal.user_id is not None:
        return "user_id = %s", (principal.user_id,)
    return "anonymous_owner_token_hash = %s", (principal.anonymous_token_hash,)


def create_chat_session(session_id: str, video_id: str, principal, title: str | None = None):
    
    owner_column = "user_id" if principal.user_id is not None else "anonymous_owner_token_hash"
    owner_value = principal.user_id or principal.anonymous_token_hash
    
    query_id="""
        SELECT id
        FROM videos
        WHERE video_id = %s;
    """
    
    
    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(query_id,(video_id,))

            video = cur.fetchone()

            if video is None:
                raise ValueError("Video does not exist")

            cur.execute(
                """
                INSERT INTO chat_sessions (
                    session_id,
                    video_id,
                    title,
                    {owner_column}
                )
                VALUES (%s, %s, %s, %s)
                RETURNING *;
                """.format(owner_column=owner_column),
                (session_id, video_id, title, owner_value),
            )

            session = cur.fetchone()
            conn.commit()

    return session


def get_chat_session(session_id: str, principal):
    owner_clause, owner_params = owner_filter(principal)
    
    query=f"""
        SELECT *
        FROM chat_sessions
        WHERE session_id = %s AND {owner_clause};
    """
    
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id, *owner_params))
            session= cur.fetchone()
            
    return session



def get_chat_sessions_for_video(video_id: str, principal):
    
    owner_clause, owner_params = owner_filter(principal)

    query = """
        SELECT cs.id, cs.session_id, cs.video_id, cs.title, cs.created_at, cs.updated_at
        FROM chat_sessions cs
        JOIN videos v ON cs.video_id = v.video_id
        WHERE v.video_id = %s AND {owner_clause}
        ORDER BY cs.updated_at DESC;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(query.format(owner_clause=owner_clause), (video_id, *owner_params))

            sessions = cur.fetchall()

    return sessions


def delete_chat_session(session_id: str, principal):
    owner_clause, owner_params = owner_filter(principal)
    query=f"""
        DELETE FROM chat_sessions
        WHERE session_id = %s AND {owner_clause}
        RETURNING *
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id, *owner_params))
            session = cur.fetchone()
            conn.commit()
            
    return session