from __future__ import annotations

import math
from collections import Counter

from epistemic_uq.schemas import AgreementStatistics, ExtractedAnswer, SemanticCluster
from epistemic_uq.uncertainty.semantic import SemanticAdjudicator


def normalized_entropy(masses: list[float]) -> float:
    positive = [mass for mass in masses if mass > 0.0]
    if len(positive) <= 1:
        return 0.0
    entropy = -sum(mass * math.log(mass) for mass in positive)
    return min(1.0, entropy / math.log(len(positive)))


def agreement_statistics(
    answers: tuple[ExtractedAnswer, ...],
    generation_ids: tuple[str, ...],
    adjudicator: SemanticAdjudicator,
    high_confidence: tuple[float | None, ...] | None = None,
    contradiction_threshold: float = 0.75,
) -> tuple[AgreementStatistics, tuple[SemanticCluster, ...]]:
    if not answers:
        empty = AgreementStatistics(
            lexical_agreement=0.0,
            semantic_agreement=0.0,
            dominant_mass=0.0,
            normalized_entropy=1.0,
            contradiction=False,
        )
        return empty, ()
    lexical_counts = Counter(answer.canonical for answer in answers)
    lexical_agreement = max(lexical_counts.values()) / len(answers)
    clusters = adjudicator.cluster(answers, generation_ids)
    dominant_mass = clusters[0].mass if clusters else 0.0
    entropy = normalized_entropy([cluster.mass for cluster in clusters])
    contradictions: list[tuple[str, str]] = []
    for left in range(len(answers)):
        for right in range(left + 1, len(answers)):
            left_confidence = high_confidence[left] if high_confidence else None
            right_confidence = high_confidence[right] if high_confidence else None
            confidence_ok = (
                high_confidence is None
                or (
                    left_confidence is not None
                    and right_confidence is not None
                    and left_confidence >= contradiction_threshold
                    and right_confidence >= contradiction_threshold
                )
            )
            if confidence_ok and adjudicator.contradictory(answers[left], answers[right]):
                contradictions.append((generation_ids[left], generation_ids[right]))
    statistics = AgreementStatistics(
        lexical_agreement=lexical_agreement,
        semantic_agreement=dominant_mass,
        dominant_mass=dominant_mass,
        normalized_entropy=entropy,
        contradiction=bool(contradictions),
        contradiction_pairs=tuple(contradictions),
    )
    return statistics, clusters


def cross_group_agreement(groups: tuple[tuple[ExtractedAnswer, ...], ...], adjudicator: SemanticAdjudicator) -> float | None:
    representatives = tuple(group[0] for group in groups if group)
    if len(representatives) < 2:
        return None
    equivalent_pairs = 0
    total_pairs = 0
    for left in range(len(representatives)):
        for right in range(left + 1, len(representatives)):
            total_pairs += 1
            equivalent_pairs += int(adjudicator.equivalent(representatives[left], representatives[right]))
    return equivalent_pairs / total_pairs if total_pairs else None
