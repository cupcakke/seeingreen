from __future__ import annotations

import os
import time
from typing import Any

import httpx

from epistemic_uq.backends.base import ModelBackend, build_generation
from epistemic_uq.errors import BackendError
from epistemic_uq.schemas import BackendCapabilities, ModelRequest, TokenProbability, Usage
from epistemic_uq.utils import elapsed_ms


class OpenAICompatibleHTTPBackend(ModelBackend):
    @property
    def capabilities(self) -> BackendCapabilities:
        options = self.config.options
        return BackendCapabilities(
            logprobs=bool(options.get("supports_logprobs", True)),
            seed=bool(options.get("supports_seed", True)),
            multi_sample=True,
            forced_confidence=True,
            streaming=False,
            local=False,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            value = os.getenv(self.config.api_key_env)
            if not value:
                raise BackendError(f"Missing API key environment variable {self.config.api_key_env}")
            headers["Authorization"] = f"Bearer {value}"
        headers.update({str(k): str(v) for k, v in self.config.options.get("headers", {}).items()})
        return headers

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.prompt.system:
            messages.append({"role": "system", "content": request.prompt.system})
        messages.append({"role": "user", "content": request.prompt.user})
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.seed is not None and self.capabilities.seed:
            payload["seed"] = request.seed
        if request.logprobs and self.capabilities.logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = int(self.config.options.get("top_logprobs", 5))
        payload.update(self.config.options.get("payload", {}))
        return payload

    def _extract_tokens(self, choice: dict[str, Any]) -> tuple[TokenProbability, ...]:
        content = ((choice.get("logprobs") or {}).get("content") or [])
        tokens: list[TokenProbability] = []
        cursor = 0
        for index, item in enumerate(content):
            token = str(item.get("token", ""))
            start = cursor
            cursor += len(token)
            tokens.append(
                TokenProbability(
                    token=token,
                    logprob=float(item["logprob"]),
                    position=index,
                    start_char=start,
                    end_char=cursor,
                )
            )
        return tuple(tokens)

    def _generate(self, request: ModelRequest):
        if not self.config.endpoint:
            raise BackendError("HTTP endpoint is not configured")
        started = time.perf_counter()
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(self.config.endpoint, headers=self._headers(), json=self._payload(request))
        if response.status_code >= 400:
            raise BackendError(f"HTTP backend returned {response.status_code}: {response.text[:1000]}")
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise BackendError("HTTP backend returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if isinstance(text, list):
            text = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in text)
        if not isinstance(text, str):
            raise BackendError("HTTP backend returned invalid text content")
        usage_data = data.get("usage") or {}
        prompt_tokens = int(usage_data.get("prompt_tokens", 0))
        completion_tokens = int(usage_data.get("completion_tokens", 0))
        total_tokens = int(usage_data.get("total_tokens", prompt_tokens + completion_tokens))
        pricing = self.config.pricing
        estimated_cost = (
            prompt_tokens * float(pricing.get("prompt_per_token", 0.0))
            + completion_tokens * float(pricing.get("completion_per_token", 0.0))
        )
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            currency=str(pricing.get("currency", "USD")),
        )
        return build_generation(
            request=request,
            backend_id=self.config.backend_id,
            text=text,
            latency_ms=elapsed_ms(started),
            model=str(data.get("model", request.model)),
            finish_reason=choice.get("finish_reason"),
            token_probabilities=self._extract_tokens(choice),
            usage=usage,
            raw_response=data,
            reproducibility={
                "backend_type": self.config.backend_type.value,
                "endpoint": self.config.endpoint,
                "request_parameters": self._payload(request),
                "system_fingerprint": data.get("system_fingerprint"),
            },
        )
