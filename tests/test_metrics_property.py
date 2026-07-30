import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from epistemic_uq.calibration.metrics import selective_curve


@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
)
def test_selective_curve_bounds(confidences: list[float]) -> None:
    labels = [int(value >= 0.5) for value in confidences]
    curve = selective_curve(confidences, labels)
    assert all(0.0 <= point.coverage <= 1.0 for point in curve)
    assert all(0.0 <= point.accuracy <= 1.0 for point in curve)
    assert all(0.0 <= point.risk <= 1.0 for point in curve)
