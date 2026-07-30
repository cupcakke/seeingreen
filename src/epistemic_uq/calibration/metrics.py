from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score

from epistemic_uq.schemas import CalibrationBin, MetricSummary
from epistemic_uq.utils import clamp_probability


@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float
    accuracy: float
    risk: float
    abstained: int
    answered: int


def _arrays(confidences, labels) -> tuple[np.ndarray, np.ndarray]:
    confidence_array = np.asarray(confidences, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if confidence_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("Confidences and labels must be one-dimensional")
    if confidence_array.size != label_array.size:
        raise ValueError("Confidences and labels must have equal length")
    if confidence_array.size == 0:
        raise ValueError("At least one observation is required")
    if np.any(~np.isfinite(confidence_array)):
        raise ValueError("Confidences must be finite")
    if np.any((confidence_array < 0.0) | (confidence_array > 1.0)):
        raise ValueError("Confidences must be in [0, 1]")
    if np.any((label_array != 0) & (label_array != 1)):
        raise ValueError("Labels must be binary")
    return confidence_array, label_array


def calibration_bins(
    confidences,
    labels,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "quantile",
) -> tuple[CalibrationBin, ...]:
    confidence_array, label_array = _arrays(confidences, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if strategy == "quantile":
        edges = np.quantile(confidence_array, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.maximum.accumulate(edges)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unknown binning strategy {strategy}")
    bins: list[CalibrationBin] = []
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            mask = (confidence_array >= lower) & (confidence_array < upper)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                CalibrationBin(
                    lower=lower,
                    upper=upper,
                    count=0,
                    mean_confidence=0.0,
                    empirical_accuracy=0.0,
                    absolute_gap=0.0,
                )
            )
            continue
        mean_confidence = float(confidence_array[mask].mean())
        empirical_accuracy = float(label_array[mask].mean())
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                absolute_gap=abs(mean_confidence - empirical_accuracy),
            )
        )
    return tuple(bins)


def expected_calibration_error(bins: tuple[CalibrationBin, ...]) -> float:
    total = sum(item.count for item in bins)
    if total == 0:
        return 0.0
    return sum(item.count / total * item.absolute_gap for item in bins)


def maximum_calibration_error(bins: tuple[CalibrationBin, ...]) -> float:
    populated = [item.absolute_gap for item in bins if item.count > 0]
    return max(populated, default=0.0)


def brier_score(confidences, labels) -> float:
    confidence_array, label_array = _arrays(confidences, labels)
    return float(np.mean((confidence_array - label_array) ** 2))


def negative_log_likelihood(confidences, labels, epsilon: float = 1e-12) -> float:
    confidence_array, label_array = _arrays(confidences, labels)
    clipped = np.clip(confidence_array, epsilon, 1.0 - epsilon)
    return float(-np.mean(label_array * np.log(clipped) + (1 - label_array) * np.log(1.0 - clipped)))


def selective_curve(confidences, labels) -> tuple[SelectivePoint, ...]:
    confidence_array, label_array = _arrays(confidences, labels)
    thresholds = sorted(set(float(value) for value in confidence_array), reverse=True)
    thresholds.append(0.0)
    points: list[SelectivePoint] = []
    total = len(confidence_array)
    for threshold in thresholds:
        mask = confidence_array >= threshold
        answered = int(mask.sum())
        coverage = answered / total
        accuracy = float(label_array[mask].mean()) if answered else 1.0
        risk = 1.0 - accuracy if answered else 0.0
        points.append(
            SelectivePoint(
                threshold=threshold,
                coverage=coverage,
                accuracy=accuracy,
                risk=risk,
                abstained=total - answered,
                answered=answered,
            )
        )
    unique: dict[float, SelectivePoint] = {}
    for point in points:
        unique[point.coverage] = point
    return tuple(sorted(unique.values(), key=lambda point: point.coverage))


def area_under_risk_coverage(curve: tuple[SelectivePoint, ...]) -> float:
    if len(curve) < 2:
        return 0.0
    coverage = np.asarray([point.coverage for point in curve], dtype=float)
    risk = np.asarray([point.risk for point in curve], dtype=float)
    return float(np.trapezoid(risk, coverage))


def calibration_report(
    confidences,
    labels,
    n_bins: int = 15,
    strategy: Literal["uniform", "quantile"] = "quantile",
) -> MetricSummary:
    confidence_array, label_array = _arrays(confidences, labels)
    bins = calibration_bins(confidence_array, label_array, n_bins=n_bins, strategy=strategy)
    auroc = None
    if len(np.unique(label_array)) == 2:
        auroc = float(roc_auc_score(label_array, confidence_array))
    curve = selective_curve(confidence_array, label_array)
    return MetricSummary(
        accuracy=float(label_array.mean()),
        ece=expected_calibration_error(bins),
        mce=maximum_calibration_error(bins),
        brier=brier_score(confidence_array, label_array),
        nll=negative_log_likelihood(confidence_array, label_array),
        auroc=auroc,
        aurc=area_under_risk_coverage(curve),
        bins=bins,
    )


def threshold_for_target_coverage(confidences, target_coverage: float) -> float:
    confidence_array = np.asarray(confidences, dtype=float)
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("Target coverage must be in [0, 1]")
    if confidence_array.size == 0:
        raise ValueError("At least one confidence is required")
    if target_coverage == 0.0:
        return 1.0
    quantile = max(0.0, min(1.0, 1.0 - target_coverage))
    return float(np.quantile(confidence_array, quantile, method="lower"))


def optimize_threshold(
    confidences,
    labels,
    objective: Literal["utility", "minimum_risk", "target_coverage"] = "utility",
    correct_utility: float = 1.0,
    incorrect_utility: float = -1.0,
    abstain_utility: float = 0.0,
    max_risk: float = 0.1,
    target_coverage: float = 0.8,
) -> SelectivePoint:
    curve = selective_curve(confidences, labels)
    if objective == "target_coverage":
        return min(curve, key=lambda point: (abs(point.coverage - target_coverage), -point.accuracy))
    if objective == "minimum_risk":
        feasible = [point for point in curve if point.risk <= max_risk and point.answered > 0]
        if not feasible:
            return min(curve, key=lambda point: (point.risk, -point.coverage))
        return max(feasible, key=lambda point: point.coverage)
    total = len(labels)
    label_array = np.asarray(labels, dtype=int)
    confidence_array = np.asarray(confidences, dtype=float)
    best_point = curve[0]
    best_utility = -math.inf
    for point in curve:
        mask = confidence_array >= point.threshold
        correct = int(label_array[mask].sum())
        incorrect = int(mask.sum()) - correct
        abstained = total - int(mask.sum())
        utility = correct * correct_utility + incorrect * incorrect_utility + abstained * abstain_utility
        if utility > best_utility or (math.isclose(utility, best_utility) and point.coverage > best_point.coverage):
            best_utility = utility
            best_point = point
    return best_point
