from epistemic_uq.backends.subprocess import SubprocessBackend
from epistemic_uq.schemas import ModelRequest, Prompt


def test_subprocess_backend_generation(backend_config) -> None:
    backend = SubprocessBackend(backend_config)
    request = ModelRequest(
        request_id="r",
        prompt=Prompt(user="What is 2 + 3?", template_id="t", template_version="1"),
        model=backend_config.model,
        logprobs=True,
        metadata={"task_type": "question_answering", "purpose": "baseline"},
    )
    generation = backend.generate(request)
    assert generation.text == "5"
    assert generation.token_probabilities
    confidence = backend.forced_confidence(request, generation.text)
    assert "0.995" in confidence.text
