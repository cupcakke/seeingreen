from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from epistemic_uq.errors import CalibrationError
from epistemic_uq.utils import clamp_probability


DEFAULT_FEATURES = (
    "self_report_confidence",
    "logprob_confidence",
    "truth_confidence",
    "self_consistency_confidence",
    "perturbation_stability",
    "cross_model_agreement",
    "semantic_agreement",
    "semantic_entropy_inverse",
    "contradiction_inverse",
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


@dataclass(frozen=True)
class FusionPrediction:
    probability: float
    contributions: dict[str, float]


class TransparentFusionModel:
    def __init__(self, feature_names: tuple[str, ...] = DEFAULT_FEATURES, monotonic: bool = True) -> None:
        self.feature_names = feature_names
        self.monotonic = monotonic
        self.coefficients = np.zeros(len(feature_names), dtype=float)
        self.intercept = 0.0
        self.means = np.zeros(len(feature_names), dtype=float)
        self.scales = np.ones(len(feature_names), dtype=float)
        self.missing_values = np.full(len(feature_names), 0.5, dtype=float)
        self.fitted = False

    def _matrix(self, feature_rows: list[dict[str, Any]], fit: bool) -> np.ndarray:
        matrix = np.empty((len(feature_rows), len(self.feature_names)), dtype=float)
        for row_index, row in enumerate(feature_rows):
            for column_index, name in enumerate(self.feature_names):
                value = row.get(name)
                matrix[row_index, column_index] = np.nan if value is None else float(value)
        if fit:
            for index in range(matrix.shape[1]):
                column = matrix[:, index]
                finite = column[np.isfinite(column)]
                self.missing_values[index] = float(np.median(finite)) if finite.size else 0.5
        for index in range(matrix.shape[1]):
            missing = ~np.isfinite(matrix[:, index])
            matrix[missing, index] = self.missing_values[index]
        if fit:
            self.means = matrix.mean(axis=0)
            scales = matrix.std(axis=0)
            self.scales = np.where(scales < 1e-9, 1.0, scales)
        return (matrix - self.means) / self.scales

    def fit(self, feature_rows: list[dict[str, Any]], labels: list[int], l2: float = 1e-3) -> TransparentFusionModel:
        if len(feature_rows) != len(labels) or not feature_rows:
            raise CalibrationError("Feature rows and labels must be non-empty and aligned")
        targets = np.asarray(labels, dtype=float)
        if np.any((targets != 0.0) & (targets != 1.0)) or len(np.unique(targets)) < 2:
            raise CalibrationError("Fusion labels must contain both binary classes")
        matrix = self._matrix(feature_rows, fit=True)
        initial = np.zeros(matrix.shape[1] + 1, dtype=float)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            weights = parameters[:-1]
            intercept = parameters[-1]
            probabilities = _sigmoid(matrix @ weights + intercept)
            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            loss = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
            loss += 0.5 * l2 * float(weights @ weights)
            residual = probabilities - targets
            gradient_weights = matrix.T @ residual / len(targets) + l2 * weights
            gradient_intercept = float(residual.mean())
            return float(loss), np.concatenate([gradient_weights, [gradient_intercept]])

        bounds = [(0.0, None) if self.monotonic else (None, None) for _ in self.feature_names]
        bounds.append((None, None))
        result = minimize(
            lambda parameters: objective(parameters)[0],
            initial,
            jac=lambda parameters: objective(parameters)[1],
            method="L-BFGS-B",
            bounds=bounds,
        )
        if not result.success:
            raise CalibrationError(str(result.message))
        self.coefficients = np.asarray(result.x[:-1], dtype=float)
        self.intercept = float(result.x[-1])
        self.fitted = True
        return self

    def predict(self, feature_rows: list[dict[str, Any]]) -> np.ndarray:
        if not self.fitted:
            raise CalibrationError("Fusion model is not fitted")
        matrix = self._matrix(feature_rows, fit=False)
        return _sigmoid(matrix @ self.coefficients + self.intercept)

    def explain(self, feature_row: dict[str, Any]) -> FusionPrediction:
        matrix = self._matrix([feature_row], fit=False)[0]
        contributions = {
            name: float(value * weight)
            for name, value, weight in zip(self.feature_names, matrix, self.coefficients, strict=True)
        }
        probability = float(_sigmoid(np.asarray([sum(contributions.values()) + self.intercept]))[0])
        return FusionPrediction(probability=probability, contributions=contributions)

    def parameters(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "coefficients": [float(value) for value in self.coefficients],
            "intercept": self.intercept,
            "means": [float(value) for value in self.means],
            "scales": [float(value) for value in self.scales],
            "missing_values": [float(value) for value in self.missing_values],
            "monotonic": self.monotonic,
        }


class RuleBasedFusion:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "self_report_confidence": 0.10,
            "logprob_confidence": 0.15,
            "truth_confidence": 0.15,
            "self_consistency_confidence": 0.20,
            "perturbation_stability": 0.15,
            "cross_model_agreement": 0.10,
            "semantic_agreement": 0.15,
        }
        total = sum(self.weights.values())
        if total <= 0.0:
            raise ValueError("Fusion weights must have positive mass")
        self.weights = {key: value / total for key, value in self.weights.items()}

    def predict_one(self, features: dict[str, Any]) -> FusionPrediction:
        available = {
            name: float(features[name])
            for name in self.weights
            if features.get(name) is not None
        }
        if not available:
            probability = 1.0 - float(features.get("epistemic_risk", 0.5))
            return FusionPrediction(probability=clamp_probability(probability), contributions={})
        total_weight = sum(self.weights[name] for name in available)
        contributions = {
            name: self.weights[name] / total_weight * value
            for name, value in available.items()
        }
        probability = sum(contributions.values())
        if bool(features.get("contradiction", False)):
            probability *= 0.5
        return FusionPrediction(probability=clamp_probability(probability), contributions=contributions)
