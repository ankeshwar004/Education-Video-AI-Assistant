from api.exceptions import bad_request, not_found, unprocessable
from database.queries.chat_session_query import delete_chat_session,get_chat_session,get_chat_sessions_for_video
from database.queries.videos_query import get_video
from src.memory import clear_memory, get_all_messages, initialize_memory
 
 
def create_session(session_id, video_id, principal, title=None):
    
    if not session_id:
        raise bad_request("session_id is required")
    if not video_id:
        raise bad_request("video_id is required")
 
    video = get_video(video_id)
    
    if video is None:
        raise not_found("Video not found")
    if video["status"] != "ready":
        raise unprocessable("Video is not ready")
 
    existing = get_chat_session(session_id, principal)
    
    if existing is not None:
        if existing["video_id"] != video_id:
            raise bad_request("Session belongs to a different video")
        return existing
 
    return initialize_memory(session_id, video_id, title, principal)
 
 
def get_session(session_id, principal):
    
    session = get_chat_session(session_id, principal)
    if session is None:
        raise not_found("Session not found")
    return session
 
 
def list_sessions(video_id, principal):
    
    video = get_video(video_id)
    if video is None:
        raise not_found("Video not found")
    return get_chat_sessions_for_video(video_id, principal)
 
 
def list_session_messages(session_id, principal):
    
    session = get_chat_session(session_id, principal)
    if session is None:
        raise not_found("Session not found")
    return get_all_messages(session_id)
 
 
def reset_session(session_id, principal):
    
    session = get_chat_session(session_id, principal)
    if session is None:
        raise not_found("Session not found")
    clear_memory(session_id)
    return get_chat_session(session_id, principal)
 
 
def remove_session(session_id, principal):
    
    session = get_chat_session(session_id, principal)
    if session is None:
        raise not_found("Session not found")
    return delete_chat_session(session_id, principal)
