from fastapi import APIRouter

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.get("/")
def get_videos():
    return {
        "message": "Video endpoint working"
    }