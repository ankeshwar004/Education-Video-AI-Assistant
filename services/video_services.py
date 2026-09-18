from api.exceptions import bad_request, conflict, not_found

from src.ingest import preprocess_video
from database.queries.videos_query import create_video, get_video, get_videos, delete_video, update_video
import yt_dlp


def get_video_metadata(url):
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = sanitize_video_id(info.get("id") or info.get("title"))
    title = info.get("title") or video_id
    return video_id, title



def queue_video_processing(youtube_url):
    
    if not youtube_url or not youtube_url.strip():
        raise bad_request("youtube_url is required")

    try:
        video_id, title = get_video_metadata(youtube_url)
    except Exception as exc:
        raise bad_request(f"Unable to read video URL: {exc}")

    existing_video=get_video(video_id)

    if existing_video is not None:
        if existing_video["status"] == "processing":
            raise conflict("Video is already processing")
        if existing_video["status"] == "ready":
            raise conflict("Video already exists")

        return update_video(video_id=video_id,title=title,status="processing")

    return create_video(video_id=video_id,title=title,youtube_url=youtube_url,status="processing")


def process_video_in_background(youtube_url, video_id):
    try:
        result = preprocess_video(video_path=None, url=youtube_url)

        if result["video_id"] != video_id:
            raise RuntimeError("Video ID changed during preprocessing")

        return update_video(video_id,result.get("video_path"),result.get("title") or video_id,duration=result.get("duration"),status="ready")
    
    except Exception:
        update_video(video_id=video_id, status="failed")
        raise


def list_videos():
    videos=get_videos()
    
    return videos

def get_video_by_id(video_id):
    video=get_video(video_id)
    
    if video is None:
        raise not_found("Video not found")
    
    return video


def remove_video(video_id,principal):
    video=get_video(video_id)
    
    if video is None:
        raise not_found("Video not found")
    
    if video["owner_id"] != principal.user_id:
        raise bad_request("You do not have permission to delete this video")
        
    if video["status"] == "processing":
        raise conflict("Video is not ready to be deleted")
    
    return delete_video(video_id)