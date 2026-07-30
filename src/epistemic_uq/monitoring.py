from __future__ import annotations

import logging
from collections import deque
from threading import Lock
from typing import Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from epistemic_uq.schemas import DriftSnapshot, TaskType
from epistemic_uq.utils import utc_now


MODEL_CALLS = Counter("euq_model_calls_total", "Language model calls", ["backend", "status"])
MODEL_LATENCY = Histogram("euq_model_call_latency_seconds", "Language model call latency", ["backend"])
MODEL_TOKENS = Counter("euq_model_tokens_total", "Language model tokens", ["backend", "kind"])
PARSING_FAILURES = Counter("euq_parsing_failures_total", "Parsing failures", ["parser"])
POLICY_ACTIONS = Counter("euq_policy_actions_total", "Decision policy actions", ["action", "task_type"])
CONFIDENCE_GAUGE = Gauge("euq_confidence_mean", "Recent confidence mean", ["signal", "task_type"])
DRIFT_ALARMS = Counter("euq_drift_alarms_total", "Drift alarms", ["signal", "task_type"])


logger = logging.getLogger(__name__)


def population_stability_index(reference, current, bins: int = 10, epsilon: float = 1e-6) -> float:
    reference_array = np.asarray(reference, dtype=float)
    current_array = np.asarray(current, dtype=float)
    if reference_array.size == 0 or current_array.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    reference_counts, _ = np.histogram(reference_array, bins=edges)
    current_counts, _ = np.histogram(current_array, bins=edges)
    reference_distribution = reference_counts / reference_counts.sum()
    current_distribution = current_counts / current_counts.sum()
    reference_distribution = np.clip(reference_distribution, epsilon, None)
    current_distribution = np.clip(current_distribution, epsilon, None)
    return float(np.sum((current_distribution - reference_distribution) * np.log(current_distribution / reference_distribution)))


class DriftMonitor:
    def __init__(
        self,
        baseline_window: int = 1000,
        current_window: int = 200,
        psi_threshold: float = 0.2,
        calibration_threshold: float = 0.08,
    ) -> None:
        if current_window >= baseline_window:
            raise ValueError("Current window must be smaller than baseline window")
        self.baseline_window = baseline_window
        self.current_window = current_window
        self.psi_threshold = psi_threshold
        self.calibration_threshold = calibration_threshold
        self._values: dict[tuple[str, str, tuple[tuple[str, str], ...]], deque[tuple[float, int | None]]] = {}
        self._lock = Lock()

    def observe(
        self,
        signal: str,
        value: float,
        task_type: TaskType | None = None,
        subgroup: dict[str, str] | None = None,
        label: int | None = None,
    ) -> DriftSnapshot | None:
        key = (signal, task_type.value if task_type else "all", tuple(sorted((subgroup or {}).items())))
        maximum = self.baseline_window + self.current_window
        with self._lock:
            series = self._values.setdefault(key, deque(maxlen=maximum))
            series.append((float(value), label))
            if len(series) < maximum:
                return None
            values = list(series)
        reference = np.asarray([item[0] for item in values[: self.baseline_window]], dtype=float)
        current = np.asarray([item[0] for item in values[-self.current_window :]], dtype=float)
        psi = population_stability_index(reference, current)
        mean_shift = float(current.mean() - reference.mean())
        reference_labels = [item[1] for item in values[: self.baseline_window]]
        current_labels = [item[1] for item in values[-self.current_window :]]
        calibration_shift = None
        if all(label is not None for label in reference_labels + current_labels):
            reference_error = abs(float(reference.mean()) - float(np.mean(reference_labels)))
            current_error = abs(float(current.mean()) - float(np.mean(current_labels)))
            calibration_shift = current_error - reference_error
        alarm = psi >= self.psi_threshold or (
            calibration_shift is not None and calibration_shift >= self.calibration_threshold
        )
        snapshot = DriftSnapshot(
            signal=signal,
            task_type=task_type,
            subgroup=subgroup or {},
            reference_count=len(reference),
            current_count=len(current),
            population_stability_index=psi,
            mean_shift=mean_shift,
            calibration_shift=calibration_shift,
            alarm=alarm,
            observed_at=utc_now(),
        )
        CONFIDENCE_GAUGE.labels(signal=signal, task_type=key[1]).set(float(current.mean()))
        if alarm:
            DRIFT_ALARMS.labels(signal=signal, task_type=key[1]).inc()
            logger.warning("confidence_drift_detected", extra=snapshot.model_dump(mode="json"))
        return snapshot
