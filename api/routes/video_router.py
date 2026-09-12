from fastapi import APIRouter

from api.schemas import VideoResponse, VideoCreateRequest
from services.video_services import get_video_by_id, list_videos, process_video

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.get("/",response_model=list[VideoResponse])
def get_videos():
    return list_videos()


@router.get("/{video_id}",response_model=VideoResponse)
def get_video(video_id: str):
    return get_video_by_id(video_id)



@router.post("/",response_model=VideoResponse)
def process(request: VideoCreateRequest):
    video = process_video(request.youtube_url)
    
    return video