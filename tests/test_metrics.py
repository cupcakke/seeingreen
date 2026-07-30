import math

from epistemic_uq.calibration.metrics import (
    calibration_report,
    optimize_threshold,
    selective_curve,
)


def test_perfect_calibration_metrics() -> None:
    confidences = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    report = calibration_report(confidences, labels, n_bins=2, strategy="uniform")
    assert report.ece == 0.0
    assert report.brier == 0.0
    assert report.nll < 1e-9


def test_threshold_optimization() -> None:
    point = optimize_threshold([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], objective="minimum_risk", max_risk=0.0)
    assert point.risk == 0.0
    assert point.coverage >= 0.5

