from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epistemic_uq.calibration.metrics import calibration_report, selective_curve
from epistemic_uq.calibration.subgroups import audit_subgroups, discover_worst_slices
from epistemic_uq.storage import Repository
from epistemic_uq.utils import utc_now


CONFIDENCE_SOURCES = (
    "calibrated_confidence",
    "self_report_confidence",
    "logprob_confidence",
    "truth_confidence",
    "self_consistency_confidence",
    "perturbation_stability",
    "cross_model_agreement",
    "semantic_agreement",
)


class ReportGenerator:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository
        template_root = Path(__file__).parent / "templates"
        self.environment = Environment(
            loader=FileSystemLoader(template_root),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def _completed(self, experiment_id: str) -> list[dict[str, Any]]:
        values = [record["result"] for record in self.repository.list_results(experiment_id) if record["status"] == "completed"]
        return [value for value in values if value is not None]

    def _source_value(self, result: dict[str, Any], source: str) -> float | None:
        if source == "calibrated_confidence":
            value = result.get(source)
        else:
            value = (result.get("features") or {}).get(source)
        return float(value) if value is not None else None

    def _metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            grouped[str(result["backend_id"])].append(result)
        output: dict[str, Any] = {}
        for backend_id, backend_results in grouped.items():
            backend_output: dict[str, Any] = {
                "count": len(backend_results),
                "evaluated_count": sum(result.get("evaluation") is not None for result in backend_results),
                "decision_counts": dict(Counter((result.get("decision") or {}).get("action", "none") for result in backend_results)),
                "confidence_sources": {},
            }
            for source in CONFIDENCE_SOURCES:
                pairs = [
                    (self._source_value(result, source), int(result["evaluation"]["correct"]))
                    for result in backend_results
                    if result.get("evaluation") is not None and self._source_value(result, source) is not None
                ]
                if not pairs:
                    continue
                confidences = [pair[0] for pair in pairs]
                labels = [pair[1] for pair in pairs]
                summary = calibration_report(confidences, labels, n_bins=min(15, max(1, len(pairs))))
                backend_output["confidence_sources"][source] = summary.model_dump(mode="json")
            output[backend_id] = backend_output
        return output

    def _subgroups(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            grouped[str(result["backend_id"])].append(result)
        for backend_id, backend_results in grouped.items():
            evaluated = [result for result in backend_results if result.get("evaluation") is not None]
            confidences = [float(result["calibrated_confidence"]) for result in evaluated]
            labels = [int(result["evaluation"]["correct"]) for result in evaluated]
            metadata = []
            for result in evaluated:
                audit = self.repository.artifacts.read(result["audit_reference"])
                request = audit.get("baseline_request") or {}
                metadata.append(dict((request.get("metadata") or {}).get("subgroup_metadata") or {}))
            if not metadata or not any(metadata):
                output[backend_id] = {"audits": [], "worst_slices": []}
                continue
            minimum_size = min(30, max(2, len(evaluated) // 10))
            audits = audit_subgroups(confidences, labels, metadata, minimum_size=minimum_size)
            worst = discover_worst_slices(confidences, labels, metadata, minimum_size=minimum_size)
            output[backend_id] = {
                "audits": [audit.model_dump(mode="json") for audit in audits],
                "worst_slices": list(worst),
            }
        return output

    def _contradictions(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contradictions = []
        for result in results:
            features = result.get("features") or {}
            if features.get("contradiction"):
                contradictions.append(
                    {
                        "example_id": result["example_id"],
                        "backend_id": result["backend_id"],
                        "answer": result["answer"]["raw"],
                        "calibrated_confidence": result.get("calibrated_confidence"),
                        "audit_reference": result.get("audit_reference"),
                    }
                )
        return sorted(contradictions, key=lambda item: -(item.get("calibrated_confidence") or 0.0))

    def _overconfident_errors(self, results: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
        errors = [
            {
                "example_id": result["example_id"],
                "backend_id": result["backend_id"],
                "answer": result["answer"]["raw"],
                "confidence": result["calibrated_confidence"],
                "decision": (result.get("decision") or {}).get("action"),
                "audit_reference": result["audit_reference"],
            }
            for result in results
            if result.get("evaluation") is not None and not result["evaluation"]["correct"]
        ]
        return sorted(errors, key=lambda item: -float(item["confidence"]))[:limit]

    def _plot_reliability(self, results: list[dict[str, Any]], output_dir: Path) -> list[str]:
        files: list[str] = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            grouped[str(result["backend_id"])].append(result)
        for backend_id, backend_results in grouped.items():
            pairs = [
                (float(result["calibrated_confidence"]), int(result["evaluation"]["correct"]))
                for result in backend_results
                if result.get("evaluation") is not None
            ]
            if not pairs:
                continue
            summary = calibration_report(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                n_bins=min(15, max(1, len(pairs))),
            )
            populated = [item for item in summary.bins if item.count > 0]
            figure, axis = plt.subplots(figsize=(7, 6))
            axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
            axis.plot(
                [item.mean_confidence for item in populated],
                [item.empirical_accuracy for item in populated],
                marker="o",
                label=backend_id,
            )
            axis.set_xlabel("Mean confidence")
            axis.set_ylabel("Empirical accuracy")
            axis.set_title(f"Reliability diagram: {backend_id}")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.legend()
            axis.grid(True, alpha=0.3)
            filename = f"reliability-{backend_id}.png"
            figure.tight_layout()
            figure.savefig(output_dir / filename, dpi=160)
            plt.close(figure)
            files.append(filename)
        return files

    def _plot_histograms(self, results: list[dict[str, Any]], output_dir: Path) -> list[str]:
        files: list[str] = []
        grouped: dict[str, list[float]] = defaultdict(list)
        for result in results:
            grouped[str(result["backend_id"])].append(float(result["calibrated_confidence"]))
        for backend_id, confidences in grouped.items():
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.hist(confidences, bins=np.linspace(0.0, 1.0, 21), edgecolor="black")
            axis.set_xlabel("Calibrated confidence")
            axis.set_ylabel("Count")
            axis.set_title(f"Confidence distribution: {backend_id}")
            filename = f"confidence-histogram-{backend_id}.png"
            figure.tight_layout()
            figure.savefig(output_dir / filename, dpi=160)
            plt.close(figure)
            files.append(filename)
        return files

    def _plot_selective(self, results: list[dict[str, Any]], output_dir: Path) -> list[str]:
        files: list[str] = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            grouped[str(result["backend_id"])].append(result)
        for backend_id, backend_results in grouped.items():
            evaluated = [result for result in backend_results if result.get("evaluation") is not None]
            if not evaluated:
                continue
            curve = selective_curve(
                [float(result["calibrated_confidence"]) for result in evaluated],
                [int(result["evaluation"]["correct"]) for result in evaluated],
            )
            figure, axis = plt.subplots(figsize=(7, 5))
            axis.plot([point.coverage for point in curve], [point.accuracy for point in curve], marker=".")
            axis.set_xlabel("Coverage")
            axis.set_ylabel("Accuracy")
            axis.set_title(f"Coverage versus accuracy: {backend_id}")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.3)
            filename = f"coverage-accuracy-{backend_id}.png"
            figure.tight_layout()
            figure.savefig(output_dir / filename, dpi=160)
            plt.close(figure)
            files.append(filename)
        return files

    def generate(self, experiment_id: str, output_dir: Path) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        run = self.repository.get_experiment(experiment_id)
        results = self._completed(experiment_id)
        metrics = self._metrics(results)
        subgroup_analysis = self._subgroups(results)
        contradictions = self._contradictions(results)
        overconfident_errors = self._overconfident_errors(results)
        images = {
            "reliability": self._plot_reliability(results, output_dir),
            "histograms": self._plot_histograms(results, output_dir),
            "selective": self._plot_selective(results, output_dir),
        }
        report = {
            "schema_version": 1,
            "generated_at": utc_now().isoformat(),
            "experiment": run.model_dump(mode="json"),
            "result_count": len(results),
            "metrics": metrics,
            "subgroups": subgroup_analysis,
            "contradictions": contradictions,
            "overconfident_errors": overconfident_errors,
            "images": images,
        }
        json_path = output_dir / "report.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        template = self.environment.get_template("report.html.j2")
        html_path = output_dir / "report.html"
        html_path.write_text(template.render(report=report), encoding="utf-8")
        return {"json": str(json_path), "html": str(html_path)}
