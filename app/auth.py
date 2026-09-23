import hashlib
import hmac
from dataclasses import dataclass
from typing import Literal

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
from starlette.requests import HTTPConnection

from app.settings import ApiKey, settings


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: Literal["admin", "judge"]


class AuthError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ConnectionHTTPBearer(HTTPBearer):
    """HTTPBearer for both HTTP and WebSocket connections.

    FastAPI's ``HTTPBearer`` is typed for ``Request``. A WebSocket upgrade is
    an ``HTTPConnection`` with the same ``Authorization`` header.
    """

    async def __call__(self, request: HTTPConnection) -> HTTPAuthorizationCredentials | None:
        authorization = request.headers.get("Authorization")
        scheme, credentials = get_authorization_scheme_param(authorization)
        if not (authorization and scheme and credentials):
            return None
        if scheme.lower() != "bearer":
            return None
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)


bearer = ConnectionHTTPBearer(auto_error=False)


def authenticate(credentials: HTTPAuthorizationCredentials | None) -> Principal:
    if credentials is None:
        raise AuthError("missing_key")
    digest = hashlib.sha256(credentials.credentials.encode("utf-8")).hexdigest()
    for record in settings.API_KEYS:
        if len(record.key_hash) == len(digest) and hmac.compare_digest(
            record.key_hash.lower(), digest
        ):
            return Principal(user_id=record.user_id, role=record.role)
    raise AuthError("unknown_key")


def key_for_user(user_id: str) -> ApiKey:
    for record in settings.API_KEYS:
        if record.user_id == user_id:
            return record
    raise AuthError("unknown_key")
