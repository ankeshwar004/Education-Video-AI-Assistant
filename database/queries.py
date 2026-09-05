from database.connection import pool


async def create_video(video_id: str,title: str | None = None,youtube_url: str | None = None,blob_url: str | None = None,duration: int | None = None):

    query = """
        INSERT INTO videos (
            video_id,
            title,
            youtube_url,
            blob_url,
            duration,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING *;
    """

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                query,
                (
                    video_id,
                    title,
                    youtube_url,
                    blob_url,
                    duration,
                    "processing",
                ),
            )

            video = await cur.fetchone()

        await conn.commit()

    return video



async def get_video(video_id: str):

    query = """
        SELECT *
        FROM videos
        WHERE video_id = %s;
    """

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query,(video_id,),)

            return await cur.fetchone()
        
        
        
        
async def update_video_status(
    video_id: str,
    status: str,
):

    query = """
        UPDATE videos
        SET status = %s
        WHERE video_id = %s
        RETURNING *;
    """

    async with pool.connection() as conn:

        async with conn.cursor() as cur:

            await cur.execute(
                query,
                (
                    status,
                    video_id,
                ),
            )

            video = await cur.fetchone()

        await conn.commit()

    return video




async def get_videos():

    query = """
        SELECT *
        FROM videos
        ORDER BY created_at DESC;
    """

    async with pool.connection() as conn:

        async with conn.cursor() as cur:

            await cur.execute(query)

            return await cur.fetchall()
        
   
   
   
        
async def delete_video(video_id: str):

    query = """
        DELETE FROM videos
        WHERE video_id = %s
        RETURNING *;
    """

    async with pool.connection() as conn:

        async with conn.cursor() as cur:

            await cur.execute(
                query,
                (video_id,),
            )

            video = await cur.fetchone()

        await conn.commit()

    return video