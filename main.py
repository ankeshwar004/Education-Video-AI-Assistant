from fastapi import FastAPI
from contextlib import asynccontextmanager
from database.connection import init_db, close_db, create_tables, check_pool
from api.routes.video_router import router as video_router
from api.routes.chat_router import router as chat_router

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

app.include_router(video_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "education-video-ai-assistant"
    }
