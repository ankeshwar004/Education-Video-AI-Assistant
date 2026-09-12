from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime



class VideoCreateRequest(BaseModel):
    youtube_url: str


class VideoResponse(BaseModel):
    video_id: str
    title: str | None = None
    youtube_url: str | None = None
    storage_path_url: str | None = None
    duration: int | None = None
    status: str
    created_at: datetime


class ChatRequest(BaseModel):
    question: str
    video_id: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    response: str
    timestamps: list[str] = Field(default_factory=list)
    source: Literal["video","general_knowledge","hybrid"]
    key_takeway: list[str]
    
    
class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime