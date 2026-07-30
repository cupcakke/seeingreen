import numpy as np
import pytest

from epistemic_uq.calibration.calibrators import CalibratorRegistry, calibrator_from_artifact
from epistemic_uq.calibration.fusion import RuleBasedFusion, TransparentFusionModel


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic", "beta"])
def test_calibrator_fit_and_roundtrip(method: str) -> None:
    confidences = [0.05, 0.15, 0.35, 0.65, 0.85, 0.95]
    labels = [0, 0, 0, 1, 1, 1]
    calibrator = CalibratorRegistry.create(method).fit(confidences, labels)
    predictions = calibrator.predict(confidences)
    restored = calibrator_from_artifact({"method": method, "parameters": calibrator.parameters()})
    restored_predictions = restored.predict(confidences)
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))
    assert np.allclose(predictions, restored_predictions, atol=1e-6)


def test_transparent_fusion_monotonic_coefficients() -> None:
    rows = [
        {"self_report_confidence": 0.1, "semantic_agreement": 0.2},
        {"self_report_confidence": 0.2, "semantic_agreement": 0.3},
        {"self_report_confidence": 0.8, "semantic_agreement": 0.7},
        {"self_report_confidence": 0.9, "semantic_agreement": 0.9},
    ]
    labels = [0, 0, 1, 1]
    model = TransparentFusionModel(feature_names=("self_report_confidence", "semantic_agreement"), monotonic=True)
    model.fit(rows, labels)
    assert np.all(model.coefficients >= 0.0)
    prediction = model.explain(rows[-1])
    assert 0.0 <= prediction.probability <= 1.0


def test_rule_fusion_penalizes_contradiction() -> None:
    model = RuleBasedFusion({"self_report_confidence": 1.0})
    normal = model.predict_one({"self_report_confidence": 0.8, "contradiction": False})
    contradiction = model.predict_one({"self_report_confidence": 0.8, "contradiction": True})
    assert contradiction.probability < normal.probability
