from datetime import datetime

from pydantic import BaseModel, Field


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=200)


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class AuthPrincipal(BaseModel):
    user_id: int | None = None
    username: str | None = None
    anonymous_token: str | None = None
    anonymous_token_hash: str | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.user_id is not None
