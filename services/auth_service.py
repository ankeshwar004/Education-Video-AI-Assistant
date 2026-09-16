import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

import config
from api.auth_schema import AuthPrincipal
from api.exceptions import bad_request, conflict
from database.queries.user_query import create_user, get_user_by_id, get_user_by_username


def normalize_username(username):
    return username.strip().casefold()


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password,encoded):
    try:
        algorithm, salt_hex, digest_hex = encoded.split("$", 2)
        if algorithm != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def register(username,password):
    normalized = normalize_username(username)
    if not normalized:
        raise bad_request("username is required")
    if get_user_by_username(normalized) is not None:
        raise conflict("Username is already registered")
    try:
        return create_user(username.strip(), normalized, hash_password(password))
    except Exception as exc:
        if "unique" in (exc).lower():
            raise conflict("Username is already registered") from exc
        raise


def authenticate(username,password):
    user = get_user_by_username(normalize_username(username))
    if user is None or not verify_password(password, user["password_hash"]):
        raise bad_request("Invalid username or password")
    return user


def create_access_token(user): 
    expires = datetime.now(timezone.utc) + timedelta(minutes=config.JWT_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": str(user["id"]), "username": user["username"], "exp": expires},
        config.JWT_SECRET_KEY,
        algorithm=config.JWT_ALGORITHM,
    )


def principal_from_token(token):
    try:
        payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
        user_id = int(payload["sub"])
        if get_user_by_id(user_id) is None:
            raise ValueError("Invalid or expired token")
        return AuthPrincipal(user_id=user_id, username=payload.get("username"))
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid or expired token") from exc


def new_anonymous_principal(token=None):
    token = token or secrets.token_urlsafe(32)
    return AuthPrincipal(
        anonymous_token=token,
        anonymous_token_hash=hashlib.sha256(token.encode()).hexdigest(),
    )