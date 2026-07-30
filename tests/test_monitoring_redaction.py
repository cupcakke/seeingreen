from epistemic_uq.monitoring import DriftMonitor, population_stability_index
from epistemic_uq.redaction import Redactor
from epistemic_uq.schemas import TaskType


def test_redaction() -> None:
    redactor = Redactor({"email", "phone"})
    value = redactor.redact("Email a@example.com or call +36 30 123 4567")
    assert "a@example.com" not in value
    assert "+36 30 123 4567" not in value


def test_population_stability_index() -> None:
    assert population_stability_index([0.1, 0.2, 0.3], [0.8, 0.9, 1.0]) > 0.0


def test_drift_monitor_emits_snapshot() -> None:
    monitor = DriftMonitor(baseline_window=4, current_window=2, psi_threshold=0.01)
    snapshot = None
    for value in [0.1, 0.1, 0.1, 0.1, 0.9, 0.9]:
        snapshot = monitor.observe("confidence", value, TaskType.CLASSIFICATION, {"group": "a"}, 1)
    assert snapshot is not None
    assert snapshot.alarm
