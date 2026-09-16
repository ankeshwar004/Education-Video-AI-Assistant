from api.exceptions import bad_request, not_found, unprocessable

from src.chat import lcel_chat
from src.memory import initialize_memory

from services.retrieval_service import load_retrieval_components
from database.queries.videos_query import get_video


def ask_question(question, video_id, session_id, principal):
    
    if not question or not question.strip():
        raise bad_request("question is required")
    if not video_id:
        raise bad_request("video_id is required")
    if not session_id:
        raise bad_request("session_id is required")
 
    video = get_video(video_id)
    if video is None:
        raise not_found("Video not found")
    if video["status"] == "processing":
        raise unprocessable("Video is still processing")
    if video["status"] == "failed":
        raise unprocessable("Video ingest failed. Re-submit the video.")
    if video["status"] != "ready":
        raise unprocessable("Video is not ready")
        
    retrieval_components = load_retrieval_components(video_id)

    initialize_memory(session_id, video_id, principal=principal)

    response = lcel_chat(question,retrieval_components,session_id)

    return response