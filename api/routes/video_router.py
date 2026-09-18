from fastapi import APIRouter, BackgroundTasks, Depends

from api.auth_dependencies import optional_principal, required_principal
from api.auth_schema import AuthPrincipal
from services.session_service import list_sessions
from api.schemas import VideoResponse, VideoCreateRequest, SessionResponse
from services.video_services import get_video_by_id, list_videos, queue_video_processing, process_video_in_background, remove_video

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.get("/",response_model=list[VideoResponse])
def get_videos():
    return list_videos()


@router.get("/{video_id}",response_model=VideoResponse)
def get_video(video_id):
    return get_video_by_id(video_id)



@router.post("/", response_model=VideoResponse, status_code=202)
def process(request: VideoCreateRequest, background_tasks: BackgroundTasks):
    video = queue_video_processing(request.youtube_url)
    background_tasks.add_task(
        process_video_in_background,
        request.youtube_url,
        video["video_id"],
    )
    return video

 
@router.get("/{video_id}/sessions",response_model=list[SessionResponse])
def get_video_sessions(video_id, principal: AuthPrincipal = Depends(optional_principal)):
    return list_sessions(video_id, principal)


@router.delete("/{video_id}", response_model=VideoResponse)
def delete_video(video_id, principal: AuthPrincipal = Depends(required_principal)):
    return remove_video(video_id,principal)