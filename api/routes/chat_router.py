from fastapi import APIRouter
from api.schemas import ChatRequest, ChatResponse, MessageResponse
from services.chat_service import ask_question
from src.memory import get_all_messages as get_session_messages

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):

    answer = ask_question(request.question,request.video_id,request.session_id)

    return ChatResponse(
        answer=answer.response,
        response=answer.response,
        timestamps=answer.timestamps,
        source=answer.source,
        key_takeway=answer.key_takeway,
    )


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
def get_messages(session_id: str):
    return get_session_messages(session_id)