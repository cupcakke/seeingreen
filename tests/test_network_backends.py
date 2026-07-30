from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from epistemic_uq.backends.http import OpenAICompatibleHTTPBackend
from epistemic_uq.backends.huggingface import HuggingFaceBackend
from epistemic_uq.backends.ollama import OllamaBackend
from epistemic_uq.errors import BackendError
from epistemic_uq.schemas import BackendConfig, ModelRequest, Prompt


class ProtocolHandler(BaseHTTPRequestHandler):
    response_kind = "http"
    status_code = 200

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if self.status_code >= 400:
            body = json.dumps({"error": "rejected"}).encode("utf-8")
            self.send_response(self.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.response_kind == "ollama":
            payload = {
                "model": request["model"],
                "message": {"role": "assistant", "content": "42"},
                "done": True,
                "prompt_eval_count": 11,
                "eval_count": 1,
                "created_at": "2026-01-01T00:00:00Z",
            }
        else:
            payload = {
                "model": request["model"],
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "42"},
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {"token": "42", "logprob": -0.01}
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
                "system_fingerprint": "fixture-v1",
            }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


@pytest.fixture
def protocol_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ProtocolHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request() -> ModelRequest:
    return ModelRequest(
        request_id="network",
        prompt=Prompt(system="system", user="What is 21 + 21?", template_id="t", template_version="1"),
        model="model",
        temperature=0.2,
        top_p=0.9,
        max_tokens=12,
        seed=7,
        stop=("stop",),
        logprobs=True,
    )


def test_http_backend_real_server(protocol_server) -> None:
    ProtocolHandler.response_kind = "http"
    ProtocolHandler.status_code = 200
    port = protocol_server.server_address[1]
    config = BackendConfig(
        backend_id="http",
        backend_type="http",
        model="model",
        endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
        timeout_seconds=5,
        retries=0,
        pricing={"prompt_per_token": 0.001, "completion_per_token": 0.002, "currency": "USD"},
        options={"supports_logprobs": True, "supports_seed": True, "headers": {"X-Test": "yes"}},
    )
    generation = OpenAICompatibleHTTPBackend(config).generate(request())
    assert generation.text == "42"
    assert generation.token_probabilities[0].token == "42"
    assert generation.usage.estimated_cost == pytest.approx(0.013)
    assert generation.reproducibility["system_fingerprint"] == "fixture-v1"


def test_http_backend_error(protocol_server) -> None:
    ProtocolHandler.status_code = 429
    port = protocol_server.server_address[1]
    config = BackendConfig(
        backend_id="http-error",
        backend_type="http",
        model="model",
        endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
        retries=0,
    )
    with pytest.raises(BackendError):
        OpenAICompatibleHTTPBackend(config).generate(request())
    ProtocolHandler.status_code = 200


def test_ollama_backend_real_server(protocol_server) -> None:
    ProtocolHandler.response_kind = "ollama"
    ProtocolHandler.status_code = 200
    port = protocol_server.server_address[1]
    config = BackendConfig(
        backend_id="ollama",
        backend_type="ollama",
        model="model",
        endpoint=f"http://127.0.0.1:{port}/api/chat",
        retries=0,
        options={"ollama_options": {"num_ctx": 1024}},
    )
    generation = OllamaBackend(config).generate(request())
    assert generation.text == "42"
    assert generation.usage.total_tokens == 12
    assert generation.reproducibility["request_parameters"]["options"]["seed"] == 7


def test_huggingface_backend_dependency_error() -> None:
    config = BackendConfig(
        backend_id="hf",
        backend_type="huggingface",
        model="local-model",
        retries=0,
    )
    backend = HuggingFaceBackend(config)
    assert backend.capabilities.local
    with pytest.raises(BackendError):
        backend.generate(request())
