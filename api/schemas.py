from pydantic import BaseModel


class VideoRequest(BaseModel):
    youtube_url: str
    video_id: str


class ChatRequest(BaseModel):
    question: str
    video_id: str


class ChatResponse(BaseModel):
    answer: str