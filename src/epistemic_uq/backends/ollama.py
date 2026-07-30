from __future__ import annotations

import time

import httpx

from epistemic_uq.backends.base import ModelBackend, build_generation
from epistemic_uq.errors import BackendError
from epistemic_uq.schemas import BackendCapabilities, ModelRequest, Usage
from epistemic_uq.utils import elapsed_ms


class OllamaBackend(ModelBackend):
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            logprobs=False,
            seed=True,
            multi_sample=True,
            forced_confidence=True,
            streaming=False,
            local=True,
        )

    def _generate(self, request: ModelRequest):
        if not self.config.endpoint:
            raise BackendError("Ollama endpoint is not configured")
        messages = []
        if request.prompt.system:
            messages.append({"role": "system", "content": request.prompt.system})
        messages.append({"role": "user", "content": request.prompt.user})
        options = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        options.update(self.config.options.get("ollama_options", {}))
        payload = {"model": request.model, "messages": messages, "stream": False, "options": options}
        started = time.perf_counter()
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.post(self.config.endpoint, json=payload)
        if response.status_code >= 400:
            raise BackendError(f"Ollama returned {response.status_code}: {response.text[:1000]}")
        data = response.json()
        message = data.get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            raise BackendError("Ollama returned invalid message content")
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost=0.0,
        )
        return build_generation(
            request=request,
            backend_id=self.config.backend_id,
            text=text,
            latency_ms=elapsed_ms(started),
            model=str(data.get("model", request.model)),
            finish_reason="stop" if data.get("done") else None,
            usage=usage,
            raw_response=data,
            reproducibility={
                "backend_type": self.config.backend_type.value,
                "endpoint": self.config.endpoint,
                "request_parameters": payload,
                "model_digest": data.get("model_digest"),
                "created_at": data.get("created_at"),
            },
        )
