from src.ingest import preprocess_video
from database.queries.videos_query import create_video,get_video,get_videos


def process_video(youtube_url):
    
    result=preprocess_video(video_path=None,url=youtube_url)
    
    video_id=result['video_id']
    
    existing_video=get_video(video_id)
    
    if existing_video is None:
        existing_video = create_video(
            video_id=video_id,
            title=video_id,
            youtube_url=youtube_url,
            storage_path_url=result["video_path"],
            status="ready",
        )

    return existing_video


def list_videos():
    videos=get_videos()
    
    return videos

def get_video_by_id(video_id):
    video=get_video(video_id)
    
    return video