from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from epistemic_uq.errors import CalibrationError
from epistemic_uq.utils import clamp_probability


def _validate(confidences, labels=None) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(confidences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise CalibrationError("Confidences must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise CalibrationError("Confidences must be finite and in [0, 1]")
    if labels is None:
        return values, None
    targets = np.asarray(labels, dtype=int)
    if targets.shape != values.shape or np.any((targets != 0) & (targets != 1)):
        raise CalibrationError("Labels must align and be binary")
    if len(np.unique(targets)) < 2:
        raise CalibrationError("Calibration fitting requires both classes")
    return values, targets


def _logit(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


class ConfidenceCalibrator(ABC):
    method: str

    @abstractmethod
    def fit(self, confidences, labels) -> ConfidenceCalibrator:
        raise NotImplementedError

    @abstractmethod
    def predict(self, confidences) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        raise NotImplementedError


class TemperatureScaling(ConfidenceCalibrator):
    method = "temperature"

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, confidences, labels) -> TemperatureScaling:
        values, targets = _validate(confidences, labels)
        logits = _logit(values)

        def objective(parameter: np.ndarray) -> float:
            temperature = math.exp(float(parameter[0]))
            probabilities = _sigmoid(logits / temperature)
            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1.0 - clipped)))

        result = minimize(objective, np.array([0.0]), method="L-BFGS-B", bounds=[(-8.0, 8.0)])
        if not result.success:
            raise CalibrationError(str(result.message))
        self.temperature = math.exp(float(result.x[0]))
        return self

    def predict(self, confidences) -> np.ndarray:
        values, _ = _validate(confidences)
        return _sigmoid(_logit(values) / self.temperature)

    def parameters(self) -> dict[str, Any]:
        return {"temperature": self.temperature}


class PlattScaling(ConfidenceCalibrator):
    method = "platt"

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)

    def fit(self, confidences, labels) -> PlattScaling:
        values, targets = _validate(confidences, labels)
        self.model.fit(_logit(values).reshape(-1, 1), targets)
        return self

    def predict(self, confidences) -> np.ndarray:
        values, _ = _validate(confidences)
        return self.model.predict_proba(_logit(values).reshape(-1, 1))[:, 1]

    def parameters(self) -> dict[str, Any]:
        return {
            "coefficient": float(self.model.coef_[0, 0]),
            "intercept": float(self.model.intercept_[0]),
        }


class IsotonicCalibration(ConfidenceCalibrator):
    method = "isotonic"

    def __init__(self) -> None:
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, confidences, labels) -> IsotonicCalibration:
        values, targets = _validate(confidences, labels)
        self.model.fit(values, targets)
        return self

    def predict(self, confidences) -> np.ndarray:
        values, _ = _validate(confidences)
        return np.asarray(self.model.predict(values), dtype=float)

    def parameters(self) -> dict[str, Any]:
        return {
            "x_thresholds": [float(value) for value in self.model.X_thresholds_],
            "y_thresholds": [float(value) for value in self.model.y_thresholds_],
        }


class BetaCalibration(ConfidenceCalibrator):
    method = "beta"

    def __init__(self) -> None:
        self.parameters_vector = np.array([1.0, 1.0, 0.0], dtype=float)

    def fit(self, confidences, labels) -> BetaCalibration:
        values, targets = _validate(confidences, labels)
        clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
        features = np.column_stack([np.log(clipped), -np.log(1.0 - clipped), np.ones_like(clipped)])

        def objective(parameters: np.ndarray) -> float:
            probabilities = _sigmoid(features @ parameters)
            probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            nll = -np.mean(targets * np.log(probabilities) + (1 - targets) * np.log(1.0 - probabilities))
            penalty = 1e-6 * float(parameters @ parameters)
            return float(nll + penalty)

        result = minimize(
            objective,
            self.parameters_vector,
            method="L-BFGS-B",
            bounds=[(0.0, None), (0.0, None), (None, None)],
        )
        if not result.success:
            raise CalibrationError(str(result.message))
        self.parameters_vector = np.asarray(result.x, dtype=float)
        return self

    def predict(self, confidences) -> np.ndarray:
        values, _ = _validate(confidences)
        clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
        features = np.column_stack([np.log(clipped), -np.log(1.0 - clipped), np.ones_like(clipped)])
        return _sigmoid(features @ self.parameters_vector)

    def parameters(self) -> dict[str, Any]:
        return {
            "a": float(self.parameters_vector[0]),
            "b": float(self.parameters_vector[1]),
            "c": float(self.parameters_vector[2]),
        }


class CalibratorRegistry:
    constructors = {
        "temperature": TemperatureScaling,
        "platt": PlattScaling,
        "isotonic": IsotonicCalibration,
        "beta": BetaCalibration,
    }

    @classmethod
    def create(cls, method: str) -> ConfidenceCalibrator:
        try:
            return cls.constructors[method]()
        except KeyError as exc:
            raise CalibrationError(f"Unknown calibrator {method}") from exc

class ParameterCalibrator:
    def __init__(self, method: str, parameters: dict[str, Any]) -> None:
        self.method = method
        self._parameters = parameters

    def predict(self, confidences) -> np.ndarray:
        values, _ = _validate(confidences)
        if self.method == "temperature":
            return _sigmoid(_logit(values) / float(self._parameters["temperature"]))
        if self.method == "platt":
            return _sigmoid(
                _logit(values) * float(self._parameters["coefficient"])
                + float(self._parameters["intercept"])
            )
        if self.method == "beta":
            clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
            linear = (
                float(self._parameters["a"]) * np.log(clipped)
                + float(self._parameters["b"]) * -np.log(1.0 - clipped)
                + float(self._parameters["c"])
            )
            return _sigmoid(linear)
        if self.method == "isotonic":
            x = np.asarray(self._parameters["x_thresholds"], dtype=float)
            y = np.asarray(self._parameters["y_thresholds"], dtype=float)
            return np.interp(values, x, y, left=y[0], right=y[-1])
        raise CalibrationError(f"Unknown parameter calibrator {self.method}")

    def parameters(self) -> dict[str, Any]:
        return dict(self._parameters)


def calibrator_from_artifact(artifact: dict[str, Any]) -> ParameterCalibrator:
    method = str(artifact["method"])
    parameters = dict(artifact["parameters"])
    return ParameterCalibrator(method, parameters)
