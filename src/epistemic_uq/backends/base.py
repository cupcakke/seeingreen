from __future__ import annotations

import abc
import logging
import time
from threading import BoundedSemaphore
from collections.abc import Callable
from typing import Any

from epistemic_uq.errors import BackendError
from epistemic_uq.schemas import BackendCapabilities, BackendConfig, Generation, ModelRequest, Prompt
from epistemic_uq.utils import elapsed_ms, trace_id, utc_now


logger = logging.getLogger(__name__)


class ModelBackend(abc.ABC):
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._semaphore = BoundedSemaphore(config.concurrency)

    @property
    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abc.abstractmethod
    def _generate(self, request: ModelRequest) -> Generation:
        raise NotImplementedError

    def generate(self, request: ModelRequest) -> Generation:
        started = time.perf_counter()
        logger.info("model_call_started", extra={"backend_id": self.config.backend_id, "request_id": request.request_id})
        generation = None
        last_error = None
        with self._semaphore:
            for attempt in range(self.config.retries + 1):
                try:
                    generation = self._generate(request)
                    break
                except (BackendError, TimeoutError, ConnectionError) as exc:
                    last_error = exc
                    logger.warning(
                        "model_call_retry",
                        extra={
                            "backend_id": self.config.backend_id,
                            "request_id": request.request_id,
                            "attempt": attempt + 1,
                            "maximum_attempts": self.config.retries + 1,
                        },
                    )
                    if attempt >= self.config.retries:
                        break
                    time.sleep(min(8.0, 2.0 ** attempt))
                except Exception as exc:
                    last_error = exc
                    break
        if generation is None:
            logger.exception(
                "model_call_failed",
                exc_info=last_error,
                extra={"backend_id": self.config.backend_id, "request_id": request.request_id},
            )
            if isinstance(last_error, BackendError):
                raise last_error
            raise BackendError(str(last_error)) from last_error
        logger.info(
            "model_call_completed",
            extra={
                "backend_id": self.config.backend_id,
                "request_id": request.request_id,
                "latency_ms": elapsed_ms(started),
                "completion_tokens": generation.usage.completion_tokens,
            },
        )
        return generation

    def sample(self, request: ModelRequest, count: int) -> tuple[Generation, ...]:
        if count < 1:
            raise ValueError("Sample count must be positive")
        generations: list[Generation] = []
        for index in range(count):
            seed = None if request.seed is None else request.seed + index
            sampled_request = request.model_copy(
                update={
                    "request_id": f"{request.request_id}:sample:{index}",
                    "seed": seed,
                    "metadata": {**request.metadata, "sample_index": index},
                }
            )
            generations.append(self.generate(sampled_request))
        return tuple(generations)

    def forced_confidence(self, source_request: ModelRequest, answer: str) -> Generation:
        confidence_prompt = Prompt(
            system="Return only strict JSON with a single numeric field confidence between 0 and 1.",
            user=(
                "Evaluate the probability that the proposed answer is correct for the original task. "
                "Do not solve a different task. Return only {\"confidence\": number}.\n\n"
                f"Original task:\n{source_request.prompt.user}\n\nProposed answer:\n{answer}"
            ),
            template_id="forced-confidence",
            template_version="1.0.0",
            variables={"answer": answer},
        )
        request = ModelRequest(
            request_id=f"{source_request.request_id}:forced-confidence",
            prompt=confidence_prompt,
            model=source_request.model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=64,
            seed=source_request.seed,
            logprobs=False,
            metadata={**source_request.metadata, "purpose": "forced_confidence"},
        )
        return self.generate(request)

    def truth_verification(self, source_request: ModelRequest, answer: str) -> Generation:
        truth_prompt = Prompt(
            system="Return only strict JSON with fields correct_probability and incorrect_probability that sum to 1.",
            user=(
                "Assess whether the proposed answer is correct for the original task. "
                "Return only JSON probabilities.\n\n"
                f"Original task:\n{source_request.prompt.user}\n\nProposed answer:\n{answer}"
            ),
            template_id="truth-verification",
            template_version="1.0.0",
            variables={"answer": answer},
        )
        request = ModelRequest(
            request_id=f"{source_request.request_id}:truth-verification",
            prompt=truth_prompt,
            model=source_request.model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=96,
            seed=source_request.seed,
            logprobs=False,
            metadata={**source_request.metadata, "purpose": "truth_verification"},
        )
        return self.generate(request)


def build_generation(
    request: ModelRequest,
    backend_id: str,
    text: str,
    latency_ms: float,
    model: str | None = None,
    finish_reason: str | None = None,
    token_probabilities=(),
    usage=None,
    raw_response: dict[str, Any] | None = None,
    reproducibility: dict[str, Any] | None = None,
) -> Generation:
    from epistemic_uq.schemas import Usage

    return Generation(
        generation_id=trace_id(),
        request_id=request.request_id,
        backend_id=backend_id,
        model=model or request.model,
        text=text,
        finish_reason=finish_reason,
        token_probabilities=tuple(token_probabilities),
        usage=usage or Usage(),
        latency_ms=latency_ms,
        created_at=utc_now(),
        raw_response=raw_response or {},
        reproducibility=reproducibility or {},
    )
