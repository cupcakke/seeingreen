from epistemic_uq.schemas import ExtractedAnswer
from epistemic_uq.uncertainty.agreement import agreement_statistics
from epistemic_uq.uncertainty.semantic import SemanticAdjudicator


def answer(value: str) -> ExtractedAnswer:
    return ExtractedAnswer(raw=value, canonical=value.casefold(), value=value, parser="test")


def test_numeric_equivalence() -> None:
    adjudicator = SemanticAdjudicator()
    assert adjudicator.equivalent(answer("1.0"), answer("1"))


def test_negation_contradiction() -> None:
    adjudicator = SemanticAdjudicator()
    assert adjudicator.contradictory(answer("yes"), answer("no"))


def test_semantic_clusters_and_entropy() -> None:
    adjudicator = SemanticAdjudicator()
    values = (answer("Yes"), answer("yes"), answer("No"))
    stats, clusters = agreement_statistics(
        values,
        ("g1", "g2", "g3"),
        adjudicator,
        high_confidence=(0.9, 0.9, 0.9),
    )
    assert len(clusters) == 2
    assert stats.dominant_mass == 2 / 3
    assert stats.contradiction
    assert 0.0 < stats.normalized_entropy <= 1.0
