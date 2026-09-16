from fastapi import APIRouter, Depends

from api.auth_dependencies import required_principal
from api.auth_schema import AuthPrincipal, AuthRequest, TokenResponse, UserResponse
from database.queries.user_query import get_user_by_id
from services.auth_service import authenticate, create_access_token, register

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
def register_user(request: AuthRequest):
    user = register(request.username, request.password)
    return TokenResponse(access_token=create_access_token(user), user=UserResponse(**user))


@router.post("/login", response_model=TokenResponse)
def login(request: AuthRequest):
    user = authenticate(request.username, request.password)
    return TokenResponse(access_token=create_access_token(user), user=UserResponse(**user))


@router.get("/me", response_model=UserResponse)
def current_user(principal: AuthPrincipal = Depends(required_principal)):
    return get_user_by_id(principal.user_id)