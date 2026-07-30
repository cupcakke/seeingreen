from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from epistemic_uq.schemas import BackendConfig, BackendType, Prompt, TokenProbability


def test_frozen_schema_rejects_mutation() -> None:
    prompt = Prompt(user="Question", template_id="t", template_version="1")
    with pytest.raises(ValidationError):
        prompt.user = "Changed"


def test_token_probability_derives_probability() -> None:
    token = TokenProbability(token="a", logprob=0.0, position=0)
    assert token.probability == 1.0


def test_http_backend_requires_endpoint() -> None:
    with pytest.raises(ValidationError):
        BackendConfig(
            backend_id="http",
            backend_type=BackendType.HTTP,
            model="model",
        )


def test_subprocess_backend_requires_command() -> None:
    with pytest.raises(ValidationError):
        BackendConfig(
            backend_id="subprocess",
            backend_type=BackendType.SUBPROCESS,
            model="model",
        )
