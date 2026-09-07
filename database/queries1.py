from database.connection import pool


def create_video(video_id: str,title: str,youtube_url: str ,storage_path_url: str | None = None,duration: int | None = None,status: str = "processing"):

    query = """
        INSERT INTO videos (
            video_id,
            title,
            youtube_url,
            storage_path_url,
            duration,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING *;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    video_id,
                    title,
                    youtube_url,
                    storage_path_url,
                    duration,
                    status,
                ),
            )

            video = cur.fetchone()

        conn.commit()

    return video



def get_video(video_id: str):

    query = """
        SELECT *
        FROM videos
        WHERE video_id = %s;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query,(video_id,),)

            return cur.fetchone()
        
        
        
        
def update_video_status(
    video_id: str,
    status: str,
):

    query = """
        UPDATE videos
        SET status = %s
        WHERE video_id = %s
        RETURNING *;
    """

    with pool.connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                (
                    status,
                    video_id,
                ),
            )

            video = cur.fetchone()

        conn.commit()

    return video




def get_videos():

    query = """
        SELECT *
        FROM videos
        ORDER BY created_at DESC;
    """

    with pool.connection() as conn:

        with conn.cursor() as cur:

            cur.execute(query)

            return cur.fetchall()
        
   
   
   
        
def delete_video(video_id: str):

    query = """
        DELETE FROM videos
        WHERE video_id = %s
        RETURNING *;
    """

    with pool.connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                (video_id,),
            )

            video = cur.fetchone()

        conn.commit()

    return video