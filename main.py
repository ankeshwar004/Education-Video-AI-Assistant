from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager
from database.connection import init_db, close_db, create_tables, check_pool
from api.routes.video_router import router as video_router
from api.routes.chat_router import router as chat_router
from src.cache import redis_client

@asynccontextmanager
async def lifespan(app: FastAPI):

    init_db()
    try:
        check_pool()
        create_tables()
        yield
        
    finally:
        close_db()



app = FastAPI(title="Education Video AI Assistant",lifespan=lifespan)



app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"])


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})



app.include_router(video_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/health")
def health_check():
    postgres_ok = False
    redis_ok = False
 
    try:
        check_pool()
        postgres_ok = True
    except Exception:
        pass
 
    try:
        redis_ok = bool(redis_client.ping())
    except Exception:
        pass
 
    if postgres_ok and redis_ok:
        return {
            "status": "healthy",
            "service": "education-video-ai-assistant",
            "postgres": postgres_ok,
            "redis": redis_ok,
        }
 
    raise HTTPException(
        status_code=503,
        detail={
            "status": "unhealthy",
            "service": "education-video-ai-assistant",
            "postgres": postgres_ok,
            "redis": redis_ok,
        },
    )