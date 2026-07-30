from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from epistemic_uq.backends.base import ModelBackend, build_generation
from epistemic_uq.errors import BackendError
from epistemic_uq.schemas import BackendCapabilities, ModelRequest, TokenProbability, Usage
from epistemic_uq.utils import elapsed_ms


class SubprocessBackend(ModelBackend):
    @property
    def capabilities(self) -> BackendCapabilities:
        options = self.config.options
        return BackendCapabilities(
            logprobs=bool(options.get("supports_logprobs", True)),
            seed=bool(options.get("supports_seed", True)),
            multi_sample=True,
            forced_confidence=True,
            streaming=False,
            local=True,
        )

    def _request_payload(self, request: ModelRequest) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "model": request.model,
            "prompt": request.prompt.model_dump(mode="json"),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "stop": list(request.stop),
            "logprobs": request.logprobs,
            "metadata": request.metadata,
        }

    def _generate(self, request: ModelRequest):
        payload = self._request_payload(request)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                list(self.config.command),
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
                env=None,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"Subprocess timed out after {self.config.timeout_seconds} seconds") from exc
        if completed.returncode != 0:
            raise BackendError(f"Subprocess failed with code {completed.returncode}: {completed.stderr[:1000]}")
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Subprocess returned invalid JSON: {completed.stdout[:1000]}") from exc
        text = data.get("text")
        if not isinstance(text, str):
            raise BackendError("Subprocess response requires string field text")
        tokens = tuple(
            TokenProbability(
                token=str(item["token"]),
                logprob=float(item["logprob"]),
                probability=float(item["probability"]) if item.get("probability") is not None else None,
                position=int(item["position"]),
                start_char=int(item["start_char"]) if item.get("start_char") is not None else None,
                end_char=int(item["end_char"]) if item.get("end_char") is not None else None,
            )
            for item in data.get("token_probabilities", [])
        )
        usage_data = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
            completion_tokens=int(usage_data.get("completion_tokens", 0)),
            total_tokens=int(usage_data.get("total_tokens", 0)),
            estimated_cost=float(usage_data.get("estimated_cost", 0.0)),
            currency=str(usage_data.get("currency", "USD")),
        )
        return build_generation(
            request=request,
            backend_id=self.config.backend_id,
            text=text,
            latency_ms=elapsed_ms(started),
            model=str(data.get("model", request.model)),
            finish_reason=data.get("finish_reason"),
            token_probabilities=tokens,
            usage=usage,
            raw_response=data,
            reproducibility={
                "backend_type": self.config.backend_type.value,
                "command": list(self.config.command),
                "request_parameters": payload,
                **dict(data.get("reproducibility") or {}),
            },
        )
