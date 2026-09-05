from fastapi import APIRouter
from api.schemas import ChatRequest, ChatResponse
from services.chat_service import ask_question

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):

    answer = ask_question(
        question=request.question,
        video_id=request.video_id
    )

    return ChatResponse(answer=answer)