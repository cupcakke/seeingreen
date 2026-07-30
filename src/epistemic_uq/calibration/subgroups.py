from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from epistemic_uq.calibration.metrics import calibration_report
from epistemic_uq.schemas import SubgroupAudit


def grouping_loss_proxy(confidences, labels, n_bins: int = 10) -> float:
    confidence_array = np.asarray(confidences, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    if confidence_array.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidence_array)
    loss = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence_array >= edges[index]) & (confidence_array <= edges[index + 1])
        else:
            mask = (confidence_array >= edges[index]) & (confidence_array < edges[index + 1])
        count = int(mask.sum())
        if count <= 1:
            continue
        residuals = confidence_array[mask] - label_array[mask]
        loss += count / total * float(np.var(residuals))
    return max(0.0, loss)


def audit_subgroups(
    confidences: list[float],
    labels: list[int],
    metadata: list[dict[str, str]],
    minimum_size: int = 30,
    n_bins: int = 10,
) -> tuple[SubgroupAudit, ...]:
    if not (len(confidences) == len(labels) == len(metadata)):
        raise ValueError("Confidences, labels and metadata must align")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, values in enumerate(metadata):
        for key, value in values.items():
            groups[(str(key), str(value))].append(index)
    audits: list[SubgroupAudit] = []
    global_accuracy = float(np.mean(labels)) if labels else 0.0
    global_confidence = float(np.mean(confidences)) if confidences else 0.0
    global_overconfidence = global_confidence - global_accuracy
    for (key, value), indices in groups.items():
        if len(indices) < minimum_size:
            continue
        subgroup_confidences = [confidences[index] for index in indices]
        subgroup_labels = [labels[index] for index in indices]
        summary = calibration_report(subgroup_confidences, subgroup_labels, n_bins=n_bins)
        mean_confidence = float(np.mean(subgroup_confidences))
        accuracy = float(np.mean(subgroup_labels))
        overconfidence = mean_confidence - accuracy
        local_loss = grouping_loss_proxy(subgroup_confidences, subgroup_labels, n_bins=n_bins)
        excess_overconfidence = max(0.0, overconfidence - global_overconfidence)
        accuracy_gap = max(0.0, global_accuracy - accuracy)
        risk_multiplier = 1.0 + excess_overconfidence + accuracy_gap + math_sqrt_safe(local_loss)
        audits.append(
            SubgroupAudit(
                subgroup_key=key,
                subgroup_value=value,
                count=len(indices),
                accuracy=accuracy,
                mean_confidence=mean_confidence,
                ece=summary.ece,
                brier=summary.brier,
                overconfidence=overconfidence,
                grouping_loss_proxy=local_loss,
                risk_multiplier=risk_multiplier,
            )
        )
    return tuple(sorted(audits, key=lambda audit: (-audit.risk_multiplier, -audit.count, audit.subgroup_key)))


def math_sqrt_safe(value: float) -> float:
    return float(np.sqrt(max(0.0, value)))


def discover_worst_slices(
    confidences: list[float],
    labels: list[int],
    metadata: list[dict[str, str]],
    minimum_size: int = 30,
    max_dimensions: int = 2,
    limit: int = 20,
) -> tuple[dict[str, Any], ...]:
    keys = sorted({key for row in metadata for key in row})
    candidates: list[dict[str, Any]] = []
    for dimension_count in range(1, min(max_dimensions, len(keys)) + 1):
        for selected_keys in combinations(keys, dimension_count):
            groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for index, row in enumerate(metadata):
                if all(key in row for key in selected_keys):
                    groups[tuple(row[key] for key in selected_keys)].append(index)
            for values, indices in groups.items():
                if len(indices) < minimum_size:
                    continue
                subgroup_confidences = [confidences[index] for index in indices]
                subgroup_labels = [labels[index] for index in indices]
                summary = calibration_report(subgroup_confidences, subgroup_labels, n_bins=min(10, len(indices)))
                overconfidence = float(np.mean(subgroup_confidences) - np.mean(subgroup_labels))
                score = max(0.0, overconfidence) + summary.ece + summary.brier
                candidates.append(
                    {
                        "slice": dict(zip(selected_keys, values, strict=True)),
                        "count": len(indices),
                        "accuracy": float(np.mean(subgroup_labels)),
                        "mean_confidence": float(np.mean(subgroup_confidences)),
                        "ece": summary.ece,
                        "brier": summary.brier,
                        "overconfidence": overconfidence,
                        "score": score,
                    }
                )
    return tuple(sorted(candidates, key=lambda item: (-item["score"], -item["count"]))[:limit])
