from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config
from api.auth_schema import AuthPrincipal
from services.auth_service import new_anonymous_principal, principal_from_token

bearer_scheme = HTTPBearer(auto_error=False)


def optional_principal(request: Request,credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
    if credentials is not None:
        try:
            principal = principal_from_token(credentials.credentials)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
    else:
        cookie = request.cookies.get(config.ANONYMOUS_OWNER_COOKIE)
        principal = new_anonymous_principal(cookie)
        
        if cookie is None:
            request.state.new_anonymous_owner = principal.anonymous_token
            
    request.state.principal = principal
    return principal


def required_principal(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return principal_from_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc