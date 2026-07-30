from __future__ import annotations

import sys
from pathlib import Path

import pytest

from epistemic_uq.backends.registry import BackendRegistry
from epistemic_uq.runner import ExperimentRunner
from epistemic_uq.schemas import BackendConfig
from epistemic_uq.storage import ArtifactStore, Repository


@pytest.fixture
def repository(tmp_path: Path) -> Repository:
    value = Repository(f"sqlite:///{tmp_path / 'test.db'}", ArtifactStore(tmp_path / "artifacts"))
    value.create_schema()
    return value


@pytest.fixture
def backend_config() -> BackendConfig:
    root = Path(__file__).resolve().parents[1]
    return BackendConfig.model_validate(
        {
            "backend_id": "reference-local",
            "backend_type": "subprocess",
            "model": "reference-engine-v1",
            "command": [sys.executable, str(root / "examples" / "local_backend.py")],
            "timeout_seconds": 10,
            "retries": 0,
            "concurrency": 2,
            "pricing": {},
            "options": {"supports_logprobs": True, "supports_seed": True},
        }
    )


@pytest.fixture
def pipeline_config() -> dict:
    return {
        "execution": {"max_workers": 2, "seed": 1729},
        "sampling": {"sample_count": 3, "temperature": 0.7, "top_p": 0.95, "max_tokens": 128},
        "perturbation": {
            "enabled": True,
            "max_variants": 2,
            "transforms": ["formatting", "task_framing"],
        },
        "uncertainty": {
            "self_report": True,
            "truth_verification": True,
            "sampling_agreement": True,
            "prompt_perturbation": True,
            "cross_model": True,
            "semantic_clustering": True,
            "contradiction_threshold": 0.75,
            "logprob_strategy": "geometric_mean",
        },
        "calibration": {"bins": 10, "strategy": "quantile"},
        "fusion": {"mode": "rule"},
        "policy": {
            "answer_threshold": 0.8,
            "warning_threshold": 0.6,
            "clarification_threshold": 0.45,
            "external_verification_threshold": 0.35,
            "criticality_adjustments": {"low": 0.0, "medium": 0.05, "high": 0.15, "critical": 0.25},
        },
        "observability": {
            "drift_window": 2,
            "baseline_window": 4,
            "psi_threshold": 0.2,
            "calibration_alarm_threshold": 0.08,
        },
    }


@pytest.fixture
def runner(repository: Repository, backend_config: BackendConfig, pipeline_config: dict) -> ExperimentRunner:
    registry = BackendRegistry()
    value = ExperimentRunner(repository, registry, pipeline_config)
    value.register_backend(backend_config)
    return value
