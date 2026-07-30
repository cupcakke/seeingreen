from fastapi.testclient import TestClient

from epistemic_uq.service import app


client = TestClient(app)
headers = {"X-API-Key": "local-development-key"}


def test_health_endpoints() -> None:
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_authentication_required() -> None:
    response = client.post(
        "/v1/thresholds/simulate",
        json={"confidences": [0.9, 0.1], "labels": [1, 0], "utilities": {}},
    )
    assert response.status_code == 401


def test_threshold_endpoint() -> None:
    response = client.post(
        "/v1/thresholds/simulate",
        headers=headers,
        json={
            "confidences": [0.95, 0.8, 0.3, 0.1],
            "labels": [1, 1, 0, 0],
            "utilities": {"objective": "minimum_risk", "max_risk": 0.0},
        },
    )
    assert response.status_code == 200
    assert response.json()["risk"] == 0.0


def test_policy_endpoint() -> None:
    response = client.post(
        "/v1/policy/decide",
        headers=headers,
        json={
            "calibrated_confidence": 0.95,
            "features": {
                "contradiction": False,
                "model_knowledge_uncertainty": 0.1,
                "prompt_sensitivity_uncertainty": 0.1,
                "decoding_instability_uncertainty": 0.1,
                "epistemic_risk": 0.1,
                "raw": {},
            },
            "criticality": "low",
            "subgroup_metadata": {},
            "subgroup_audits": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["action"] == "answer"


def test_uncertainty_batch_calibration_and_metrics(tmp_path) -> None:
    import json
    import sys
    from pathlib import Path

    import yaml

    from epistemic_uq.schemas import BackendConfig
    from epistemic_uq.service import registry, repository, settings

    root = Path(__file__).resolve().parents[1]
    backend_id = "service-reference"
    backend = BackendConfig.model_validate(
        {
            "backend_id": backend_id,
            "backend_type": "subprocess",
            "model": "reference-engine-v1",
            "command": [sys.executable, str(root / "examples" / "local_backend.py")],
            "timeout_seconds": 10,
            "retries": 0,
            "concurrency": 1,
            "pricing": {},
            "options": {"supports_logprobs": True, "supports_seed": True},
        }
    )
    repository.register_backend(backend)
    registered = {item.config.backend_id for item in registry.all()}
    if backend_id not in registered:
        registry.register(backend)
    query = client.post(
        "/v1/uncertainty",
        headers=headers,
        json={
            "backend_ids": [backend_id],
            "task": {
                "example_id": "service-math",
                "dataset_id": "service",
                "task_type": "question_answering",
                "user_input": "What is 5 + 7?",
                "expected_format": "number",
                "reference_label": 12,
                "valid_answers": [],
                "subgroup_metadata": {"domain": "arithmetic"},
                "perturbation_rules": {},
                "validator_config": {"method": "numeric"},
                "criticality": "low",
                "metadata": {},
            },
            "generation": {},
            "config_overrides": {
                "sampling": {"sample_count": 1, "max_tokens": 64},
                "perturbation": {"enabled": False, "max_variants": 0, "transforms": []},
                "uncertainty": {
                    "self_report": False,
                    "truth_verification": False,
                    "sampling_agreement": True,
                    "prompt_perturbation": False,
                },
            },
        },
    )
    assert query.status_code == 200
    assert query.json()[0]["answer"]["canonical"] == "12"
    settings.allowed_data_root = tmp_path
    dataset = tmp_path / "batch.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "example_id": "batch-math",
                "task_type": "question_answering",
                "user_input": "What is 9 + 1?",
                "expected_format": "number",
                "reference_label": 10,
                "valid_answers": [],
                "subgroup_metadata": {},
                "perturbation_rules": {},
                "validator_config": {"method": "numeric"},
                "criticality": "low",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "batch.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "execution": {"max_workers": 1, "seed": 2},
                "sampling": {"sample_count": 1, "temperature": 0.2, "top_p": 0.9, "max_tokens": 64},
                "perturbation": {"enabled": False, "max_variants": 0, "transforms": []},
                "uncertainty": {
                    "self_report": False,
                    "truth_verification": False,
                    "sampling_agreement": True,
                    "prompt_perturbation": False,
                    "contradiction_threshold": 0.75,
                },
                "fusion": {"mode": "rule"},
                "policy": {
                    "answer_threshold": 0.8,
                    "warning_threshold": 0.6,
                    "clarification_threshold": 0.45,
                    "external_verification_threshold": 0.35,
                    "criticality_adjustments": {"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0},
                },
                "observability": {
                    "drift_window": 2,
                    "baseline_window": 4,
                    "psi_threshold": 0.2,
                    "calibration_alarm_threshold": 0.08,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    batch = client.post(
        "/v1/batch/evaluate",
        headers=headers,
        json={"dataset_path": str(dataset), "backend_ids": [backend_id], "config_path": str(config)},
    )
    assert batch.status_code == 200
    assert batch.json()["status"] == "completed"
    repository.save_calibration("service-calibration", "temperature", {"method": "temperature", "parameters": {"temperature": 1.0}})
    calibration = client.get("/v1/calibrations/service-calibration", headers=headers)
    assert calibration.status_code == 200
    missing = client.get("/v1/calibrations/missing-calibration", headers=headers)
    assert missing.status_code == 404
    assert client.get("/metrics").status_code == 200
