from database.connection import pool


def create_chat_session(session_id: str, video_id: str, title: str | None = None):
    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT id
                FROM videos
                WHERE video_id = %s;
                """,
                (video_id,),
            )

            video = cur.fetchone()

            if video is None:
                raise ValueError("Video does not exist")

            cur.execute(
                """
                INSERT INTO chat_sessions (
                    session_id,
                    video_id,
                    title
                )
                VALUES (%s, %s, %s)
                RETURNING id, session_id, video_id, title, created_at, updated_at;
                """,
                (session_id, video_id, title),
            )

            session = cur.fetchone()
            conn.commit()

    return session


def get_chat_session(session_id: str):
    
    query="""
        SELECT *
        FROM chat_sessions
        WHERE session_id = %s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            session= cur.fetchone()
            
    return session



def get_chat_sessions_for_video(video_id: str):

    query = """
        SELECT cs.id, cs.session_id, cs.title, cs.created_at, cs.updated_at
        FROM chat_sessions cs
        JOIN videos v ON cs.video_id = v.video_id
        WHERE v.video_id = %s
        ORDER BY cs.updated_at DESC;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(query,(video_id,))

            sessions = cur.fetchall()

    return sessions


def delete_chat_session(session_id: str):
    
    query="""
        DELETE FROM chat_sessions
        WHERE session_id = %s
        RETURNING *
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            session = cur.fetchone()
            conn.commit()
            
    return session