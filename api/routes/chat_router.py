from fastapi import APIRouter, Depends

from api.auth_dependencies import optional_principal
from api.auth_schema import AuthPrincipal
from api.schemas import ChatRequest, ChatResponse, MessageResponse, SessionCreateRequest, SessionResponse
from services.session_service import create_session, get_session, list_session_messages, reset_session, remove_session
from services.chat_service import ask_question

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest, principal: AuthPrincipal = Depends(optional_principal)):

    answer = ask_question(request.question, request.video_id, request.session_id, principal)

    return ChatResponse(
        response=answer.response,
        timestamps=answer.timestamps,
        source=answer.source,
        key_takeway=answer.key_takeway,
    )


@router.post("/sessions", response_model=SessionResponse)
def create_chat_session(request: SessionCreateRequest, principal: AuthPrincipal = Depends(optional_principal)):
    return create_session(request.session_id, request.video_id, principal, request.title)
 
 
@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_chat_session(session_id: str, principal: AuthPrincipal = Depends(optional_principal)):
    return get_session(session_id, principal)



@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
def get_messages(session_id: str, principal: AuthPrincipal = Depends(optional_principal)):
    return list_session_messages(session_id, principal)


@router.delete("/sessions/{session_id}/messages", response_model=SessionResponse)
def clear_messages(session_id: str, principal: AuthPrincipal = Depends(optional_principal)):
    return reset_session(session_id, principal)
 
 
@router.delete("/sessions/{session_id}", response_model=SessionResponse)
def delete_session(session_id: str, principal: AuthPrincipal = Depends(optional_principal)):
    return remove_session(session_id, principal)
