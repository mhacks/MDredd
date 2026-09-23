import hashlib
import hmac
from dataclasses import dataclass
from typing import Literal

from app.settings import ApiKey, settings


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: Literal["admin", "judge"]


class AuthError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def authenticate(authorization: str | None) -> Principal:
    if authorization is None:
        raise AuthError("missing_key")
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or token == "":
        raise AuthError("missing_key")
    digest = hash_api_key(token)
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
