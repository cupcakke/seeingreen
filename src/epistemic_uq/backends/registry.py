from __future__ import annotations

from epistemic_uq.backends.base import ModelBackend
from epistemic_uq.backends.huggingface import HuggingFaceBackend
from epistemic_uq.backends.http import OpenAICompatibleHTTPBackend
from epistemic_uq.backends.ollama import OllamaBackend
from epistemic_uq.backends.subprocess import SubprocessBackend
from epistemic_uq.schemas import BackendConfig, BackendType


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, ModelBackend] = {}

    def register(self, config: BackendConfig) -> ModelBackend:
        constructors = {
            BackendType.HTTP: OpenAICompatibleHTTPBackend,
            BackendType.OLLAMA: OllamaBackend,
            BackendType.HUGGINGFACE: HuggingFaceBackend,
            BackendType.SUBPROCESS: SubprocessBackend,
        }
        backend = constructors[config.backend_type](config)
        self._backends[config.backend_id] = backend
        return backend

    def get(self, backend_id: str) -> ModelBackend:
        try:
            return self._backends[backend_id]
        except KeyError as exc:
            raise KeyError(f"Unknown backend {backend_id}") from exc

    def all(self) -> tuple[ModelBackend, ...]:
        return tuple(self._backends.values())

    def configs(self) -> tuple[BackendConfig, ...]:
        return tuple(backend.config for backend in self._backends.values())
