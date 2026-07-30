from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from epistemic_uq import cli
from epistemic_uq.settings import settings


def write_dataset(path: Path, suffix: str) -> None:
    records = [
        {
            "example_id": f"correct-{suffix}",
            "task_type": "question_answering",
            "user_input": "What is 2 + 2?",
            "expected_format": "number",
            "reference_label": 4,
            "valid_answers": [],
            "subgroup_metadata": {"group": "correct"},
            "perturbation_rules": {},
            "validator_config": {"method": "numeric"},
            "criticality": "low",
            "metadata": {},
        },
        {
            "example_id": f"incorrect-{suffix}",
            "task_type": "question_answering",
            "user_input": "Name the capital of France.",
            "expected_format": "short answer",
            "reference_label": "Paris",
            "valid_answers": [],
            "subgroup_metadata": {"group": "incorrect"},
            "perturbation_rules": {},
            "validator_config": {"method": "auto"},
            "criticality": "low",
            "metadata": {},
        },
    ]
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")


def test_cli_complete_workflow(tmp_path: Path, monkeypatch, capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    database = tmp_path / "cli.db"
    artifacts = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "execution": {"max_workers": 1, "seed": 1729},
                "sampling": {"sample_count": 1, "temperature": 0.4, "top_p": 0.9, "max_tokens": 64},
                "perturbation": {"enabled": False, "max_variants": 0, "transforms": []},
                "uncertainty": {
                    "self_report": False,
                    "truth_verification": False,
                    "sampling_agreement": True,
                    "prompt_perturbation": False,
                    "cross_model": False,
                    "semantic_clustering": True,
                    "contradiction_threshold": 0.75,
                },
                "calibration": {"bins": 4, "strategy": "uniform"},
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
    settings.database_url = f"sqlite:///{database}"
    settings.artifact_root = artifacts
    settings.default_config = config_path
    monkeypatch.chdir(root)
    backend_path = tmp_path / "backend.json"
    backend_path.write_text(
        json.dumps(
            {
                "backend_id": "cli-reference",
                "backend_type": "subprocess",
                "model": "reference-engine-v1",
                "command": [sys.executable, str(root / "examples" / "local_backend.py")],
                "timeout_seconds": 10,
                "retries": 0,
                "concurrency": 1,
                "pricing": {},
                "options": {"supports_logprobs": True, "supports_seed": True},
            }
        ),
        encoding="utf-8",
    )
    cli.backend_register(backend_path, config_path)
    cli.backend_list()
    development = tmp_path / "development.jsonl"
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    write_dataset(development, "dev")
    write_dataset(validation, "val")
    write_dataset(test, "test")
    cli.dataset_register(development, "development", "1", config_path)
    ids: list[str] = []
    for dataset, dataset_id in [(development, "development"), (validation, "validation"), (test, "test")]:
        cli.experiment_run(dataset, dataset_id, ["cli-reference"], "1", config_path, None, None)
        repository, _, _ = cli.runtime(config_path)
        ids.append(repository.list_experiments()[0].experiment_id)
    test_id, validation_id, development_id = ids[2], ids[1], ids[0]
    cli.experiment_list()
    cli.experiment_show(test_id)
    metrics_path = tmp_path / "metrics.json"
    cli.metrics_compute(test_id, "calibrated_confidence", 4, "uniform", metrics_path)
    assert metrics_path.exists()
    report_dir = tmp_path / "report"
    cli.report_generate(test_id, report_dir)
    assert (report_dir / "report.html").exists()
    cli.threshold_simulate(test_id, "calibrated_confidence", "utility", 1.0, -1.0, 0.0, 0.1, 0.5)
    cli.calibration_fit(development_id, test_id, "calibrated_confidence", "isotonic", "cli-isotonic")
    cli.fusion_fit(development_id, validation_id, test_id, "cli-fusion", True)
    repository, _, _ = cli.runtime(config_path)
    assert repository.get_calibration("cli-isotonic")["method"] == "isotonic"
    assert repository.get_calibration("cli-fusion")["method"] == "transparent_fusion"
    assert cli.load_mapping(config_path)["execution"]["max_workers"] == 1
    json_config = tmp_path / "mapping.json"
    json_config.write_text('{"a":1}', encoding="utf-8")
    assert cli.load_mapping(json_config) == {"a": 1}
    output_path = tmp_path / "output.json"
    cli.write_json({"x": 1}, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"x": 1}
    captured = capsys.readouterr()
    assert "cli-reference" in captured.out
