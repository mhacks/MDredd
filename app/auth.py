import hmac

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.settings import settings


class AuthError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


bearer = HTTPBearer(auto_error=False)


def authenticate(credentials: HTTPAuthorizationCredentials | None) -> None:
    token = settings.API_TOKEN
    if credentials is None or not token:
        raise AuthError("missing_key")
    presented = credentials.credentials.encode("utf-8")
    expected = token.encode("utf-8")
    if len(presented) != len(expected) or not hmac.compare_digest(presented, expected):
        raise AuthError("unknown_key")
