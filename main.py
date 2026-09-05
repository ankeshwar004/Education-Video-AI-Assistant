from fastapi import FastAPI
from contextlib import asynccontextmanager
from database.connection import init_db, close_db, create_tables, check_pool
from database.queries import *


@asynccontextmanager
async def lifespan(app: FastAPI):

    init_db()
    
    check_pool()

    create_tables()

    yield

    close_db()



app = FastAPI(
    title="Education Video AI Assistant",
    lifespan=lifespan,
)

# app.include_router(video_router, prefix="/api")
# app.include_router(chat_router, prefix="/api")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "education-video-ai-assistant"
    }
    
@app.get("/videos")
async def videos():

    return get_videos()
    


@app.post("/test-video")
async def test_video():

    video = create_video(
        video_id="test123",
        title="Python Introduction",
        youtube_url="https://youtube.com/test",
        duration=600,
    )

    return video