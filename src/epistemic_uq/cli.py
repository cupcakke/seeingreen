from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer
import yaml

from epistemic_uq.backends.registry import BackendRegistry
from epistemic_uq.calibration.calibrators import CalibratorRegistry
from epistemic_uq.calibration.fusion import TransparentFusionModel
from epistemic_uq.calibration.metrics import calibration_report, negative_log_likelihood, optimize_threshold
from epistemic_uq.logging import configure_logging
from epistemic_uq.reports import ReportGenerator
from epistemic_uq.runner import ExperimentRunner
from epistemic_uq.schemas import BackendConfig
from epistemic_uq.settings import settings
from epistemic_uq.storage import ArtifactStore, Repository
from epistemic_uq.utils import stable_hash, utc_now


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
dataset_app = typer.Typer(no_args_is_help=True)
backend_app = typer.Typer(no_args_is_help=True)
experiment_app = typer.Typer(no_args_is_help=True)
metrics_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
threshold_app = typer.Typer(no_args_is_help=True)
calibration_app = typer.Typer(no_args_is_help=True)
fusion_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(backend_app, name="backend")
app.add_typer(experiment_app, name="experiment")
app.add_typer(metrics_app, name="metrics")
app.add_typer(report_app, name="report")
app.add_typer(threshold_app, name="threshold")
app.add_typer(calibration_app, name="calibration")
app.add_typer(fusion_app, name="fusion")


def runtime(config_path: Path | None = None) -> tuple[Repository, BackendRegistry, ExperimentRunner]:
    configure_logging()
    repository = Repository(settings.database_url, ArtifactStore(settings.artifact_root))
    repository.create_schema()
    registry = BackendRegistry()
    config = settings.load_pipeline_config(config_path)
    runner = ExperimentRunner(repository, registry, config)
    return repository, registry, runner


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.casefold() in {".yaml", ".yml"}:
            value = yaml.safe_load(handle)
        else:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise typer.BadParameter("Configuration file must contain an object")
    return value


def write_json(value: Any, output: Path | None) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    else:
        typer.echo(content)


def completed_results(repository: Repository, experiment_id: str) -> list[dict[str, Any]]:
    return [
        record["result"]
        for record in repository.list_results(experiment_id)
        if record["status"] == "completed" and record["result"] is not None
    ]


def source_value(result: dict[str, Any], source: str) -> float | None:
    if source == "calibrated_confidence":
        value = result.get(source)
    else:
        value = (result.get("features") or {}).get(source)
    return float(value) if value is not None else None


def labeled_source_rows(repository: Repository, experiment_id: str, source: str) -> tuple[list[float], list[int]]:
    pairs = [
        (source_value(result, source), int(result["evaluation"]["correct"]))
        for result in completed_results(repository, experiment_id)
        if result.get("evaluation") is not None and source_value(result, source) is not None
    ]
    if not pairs:
        raise typer.BadParameter(f"Experiment {experiment_id} has no labeled values for source {source}")
    return [float(pair[0]) for pair in pairs], [pair[1] for pair in pairs]


def fusion_rows(repository: Repository, experiment_id: str) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    for result in completed_results(repository, experiment_id):
        if result.get("evaluation") is None:
            continue
        features = dict(result.get("features") or {})
        entropy = features.get("semantic_entropy")
        features["semantic_entropy_inverse"] = None if entropy is None else 1.0 - float(entropy)
        features["contradiction_inverse"] = 0.0 if features.get("contradiction") else 1.0
        rows.append(features)
        labels.append(int(result["evaluation"]["correct"]))
    if not rows:
        raise typer.BadParameter(f"Experiment {experiment_id} has no labeled fusion rows")
    return rows, labels


@dataset_app.command("register")
def dataset_register(
    path: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    dataset_id: str = typer.Option(...),
    version: str = typer.Option("1"),
    config: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    _, _, runner = runtime(config)
    manifest = runner.register_dataset(path, dataset_id, version)
    typer.echo(manifest.model_dump_json(indent=2))


@backend_app.command("register")
def backend_register(
    path: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    config: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    _, _, runner = runtime(config)
    backend = BackendConfig.model_validate(load_mapping(path))
    runner.register_backend(backend)
    typer.echo(backend.model_dump_json(indent=2))


@backend_app.command("list")
def backend_list() -> None:
    repository, _, _ = runtime()
    typer.echo(json.dumps([item.model_dump(mode="json") for item in repository.list_backends()], indent=2, sort_keys=True))


@experiment_app.command("list")
def experiment_list() -> None:
    repository, _, _ = runtime()
    typer.echo(json.dumps([item.model_dump(mode="json") for item in repository.list_experiments()], indent=2, sort_keys=True))


@experiment_app.command("show")
def experiment_show(experiment_id: str = typer.Option(...)) -> None:
    repository, _, _ = runtime()
    typer.echo(repository.get_experiment(experiment_id).model_dump_json(indent=2))


@experiment_app.command("run")
def experiment_run(
    dataset: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    dataset_id: str = typer.Option(...),
    backend: list[str] = typer.Option(...),
    version: str = typer.Option("1"),
    config: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
    resume: str | None = typer.Option(None),
    overrides: Path | None = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    _, _, runner = runtime(config)
    run = runner.run_dataset(
        dataset_path=dataset,
        dataset_id=dataset_id,
        backend_ids=tuple(backend),
        version=version,
        overrides=load_mapping(overrides) if overrides else None,
        resume_experiment_id=resume,
    )
    typer.echo(run.model_dump_json(indent=2))


@metrics_app.command("compute")
def metrics_compute(
    experiment_id: str = typer.Option(...),
    source: str = typer.Option("calibrated_confidence"),
    bins: int = typer.Option(15, min=1),
    strategy: str = typer.Option("quantile"),
    output: Path | None = typer.Option(None),
) -> None:
    repository, _, _ = runtime()
    confidences, labels = labeled_source_rows(repository, experiment_id, source)
    summary = calibration_report(confidences, labels, n_bins=bins, strategy=strategy)
    write_json(summary.model_dump(mode="json"), output)


@report_app.command("generate")
def report_generate(
    experiment_id: str = typer.Option(...),
    output_dir: Path = typer.Option(...),
) -> None:
    repository, _, _ = runtime()
    paths = ReportGenerator(repository).generate(experiment_id, output_dir)
    typer.echo(json.dumps(paths, indent=2, sort_keys=True))


@threshold_app.command("simulate")
def threshold_simulate(
    experiment_id: str = typer.Option(...),
    source: str = typer.Option("calibrated_confidence"),
    objective: str = typer.Option("utility"),
    correct_utility: float = typer.Option(1.0),
    incorrect_utility: float = typer.Option(-1.0),
    abstain_utility: float = typer.Option(0.0),
    max_risk: float = typer.Option(0.1, min=0.0, max=1.0),
    target_coverage: float = typer.Option(0.8, min=0.0, max=1.0),
) -> None:
    repository, _, _ = runtime()
    confidences, labels = labeled_source_rows(repository, experiment_id, source)
    point = optimize_threshold(
        confidences,
        labels,
        objective=objective,
        correct_utility=correct_utility,
        incorrect_utility=incorrect_utility,
        abstain_utility=abstain_utility,
        max_risk=max_risk,
        target_coverage=target_coverage,
    )
    typer.echo(json.dumps(point.__dict__, indent=2, sort_keys=True))


@calibration_app.command("fit")
def calibration_fit(
    development_experiment: str = typer.Option(...),
    test_experiment: str = typer.Option(...),
    source: str = typer.Option("calibrated_confidence"),
    method: str = typer.Option("isotonic"),
    calibrator_id: str = typer.Option(...),
) -> None:
    repository, _, _ = runtime()
    development_confidences, development_labels = labeled_source_rows(repository, development_experiment, source)
    test_confidences, test_labels = labeled_source_rows(repository, test_experiment, source)
    calibrator = CalibratorRegistry.create(method).fit(development_confidences, development_labels)
    development_calibrated = calibrator.predict(development_confidences)
    test_calibrated = calibrator.predict(test_confidences)
    artifact = {
        "schema_version": 1,
        "calibrator_id": calibrator_id,
        "method": method,
        "source": source,
        "parameters": calibrator.parameters(),
        "development_experiment": development_experiment,
        "test_experiment": test_experiment,
        "fitted_at": utc_now().isoformat(),
        "development_before": calibration_report(development_confidences, development_labels).model_dump(mode="json"),
        "development_after": calibration_report(development_calibrated, development_labels).model_dump(mode="json"),
        "test_before": calibration_report(test_confidences, test_labels).model_dump(mode="json"),
        "test_after": calibration_report(test_calibrated, test_labels).model_dump(mode="json"),
        "training_hash": stable_hash({"confidences": development_confidences, "labels": development_labels}),
    }
    repository.save_calibration(calibrator_id, method, artifact)
    typer.echo(json.dumps(artifact, indent=2, sort_keys=True))


@fusion_app.command("fit")
def fusion_fit(
    train_experiment: str = typer.Option(...),
    validation_experiment: str = typer.Option(...),
    test_experiment: str = typer.Option(...),
    model_id: str = typer.Option(...),
    monotonic: bool = typer.Option(True),
) -> None:
    repository, _, _ = runtime()
    train_rows, train_labels = fusion_rows(repository, train_experiment)
    validation_rows, validation_labels = fusion_rows(repository, validation_experiment)
    test_rows, test_labels = fusion_rows(repository, test_experiment)
    candidates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    selected_l2 = candidates[0]
    selected_loss = float("inf")
    for l2 in candidates:
        candidate = TransparentFusionModel(monotonic=monotonic).fit(train_rows, train_labels, l2=l2)
        validation_probabilities = candidate.predict(validation_rows)
        loss = negative_log_likelihood(validation_probabilities, validation_labels)
        if loss < selected_loss:
            selected_loss = loss
            selected_l2 = l2
    combined_rows = train_rows + validation_rows
    combined_labels = train_labels + validation_labels
    model = TransparentFusionModel(monotonic=monotonic).fit(combined_rows, combined_labels, l2=selected_l2)
    train_probabilities = model.predict(train_rows)
    validation_probabilities = model.predict(validation_rows)
    test_probabilities = model.predict(test_rows)
    artifact = {
        "schema_version": 1,
        "calibrator_id": model_id,
        "method": "transparent_fusion",
        "parameters": model.parameters(),
        "selected_l2": selected_l2,
        "train_experiment": train_experiment,
        "validation_experiment": validation_experiment,
        "test_experiment": test_experiment,
        "fitted_at": utc_now().isoformat(),
        "train_metrics": calibration_report(train_probabilities, train_labels).model_dump(mode="json"),
        "validation_metrics": calibration_report(validation_probabilities, validation_labels).model_dump(mode="json"),
        "test_metrics": calibration_report(test_probabilities, test_labels).model_dump(mode="json"),
        "training_hash": stable_hash({"train": train_experiment, "validation": validation_experiment}),
    }
    repository.save_calibration(model_id, "transparent_fusion", artifact)
    typer.echo(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
