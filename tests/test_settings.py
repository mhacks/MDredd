import pytest
from pydantic import ValidationError

from app.settings import Settings


@pytest.mark.parametrize("token", ["", "short-token"])
def test_missing_or_weak_token_is_rejected(token: str) -> None:
    with pytest.raises(ValidationError, match="MDREDD_API_TOKEN"):
        Settings.model_validate({"API_TOKEN": token})


def test_strong_token_is_accepted() -> None:
    token = "a" * 32
    assert Settings.model_validate({"API_TOKEN": token}).API_TOKEN == token
