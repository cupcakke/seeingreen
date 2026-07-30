from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from epistemic_uq.schemas import UncertaintyFeatures
from epistemic_uq.utils import clamp_probability


@dataclass(frozen=True)
class CompositeWeights:
    knowledge: float = 0.45
    prompt: float = 0.25
    decoding: float = 0.30
    contradiction_penalty: float = 0.20


class EpistemicRiskEstimator:
    def __init__(self, weights: CompositeWeights | None = None) -> None:
        self.weights = weights or CompositeWeights()

    def _mean_available(self, values: list[float | None], default: float) -> float:
        available = [float(value) for value in values if value is not None]
        return sum(available) / len(available) if available else default

    def estimate(
        self,
        self_report_confidence: float | None,
        logprob_confidence: float | None,
        truth_confidence: float | None,
        self_consistency_confidence: float | None,
        perturbation_stability: float | None,
        cross_model_agreement: float | None,
        semantic_agreement: float | None,
        semantic_entropy: float | None,
        contradiction: bool,
        calibration_residual: float | None = None,
        local_overconfidence: float | None = None,
        raw: dict[str, Any] | None = None,
    ) -> UncertaintyFeatures:
        direct_confidence = self._mean_available(
            [self_report_confidence, logprob_confidence, truth_confidence],
            default=0.5,
        )
        consistency_confidence = self._mean_available(
            [self_consistency_confidence, semantic_agreement, cross_model_agreement],
            default=direct_confidence,
        )
        model_knowledge_uncertainty = 1.0 - self._mean_available(
            [truth_confidence, cross_model_agreement, direct_confidence],
            default=0.5,
        )
        if calibration_residual is not None:
            model_knowledge_uncertainty = clamp_probability(
                model_knowledge_uncertainty + max(0.0, calibration_residual) * 0.25
            )
        if local_overconfidence is not None:
            model_knowledge_uncertainty = clamp_probability(
                model_knowledge_uncertainty + max(0.0, local_overconfidence) * 0.25
            )
        prompt_sensitivity_uncertainty = 1.0 - (
            perturbation_stability if perturbation_stability is not None else consistency_confidence
        )
        decoding_instability_uncertainty = self._mean_available(
            [
                None if self_consistency_confidence is None else 1.0 - self_consistency_confidence,
                semantic_entropy,
                None if semantic_agreement is None else 1.0 - semantic_agreement,
            ],
            default=1.0 - consistency_confidence,
        )
        epistemic_risk = (
            self.weights.knowledge * model_knowledge_uncertainty
            + self.weights.prompt * prompt_sensitivity_uncertainty
            + self.weights.decoding * decoding_instability_uncertainty
        )
        if contradiction:
            epistemic_risk += self.weights.contradiction_penalty
        epistemic_risk = clamp_probability(epistemic_risk)
        return UncertaintyFeatures(
            self_report_confidence=self_report_confidence,
            logprob_confidence=logprob_confidence,
            truth_confidence=truth_confidence,
            self_consistency_confidence=self_consistency_confidence,
            perturbation_stability=perturbation_stability,
            cross_model_agreement=cross_model_agreement,
            semantic_agreement=semantic_agreement,
            semantic_entropy=semantic_entropy,
            contradiction=contradiction,
            model_knowledge_uncertainty=clamp_probability(model_knowledge_uncertainty),
            prompt_sensitivity_uncertainty=clamp_probability(prompt_sensitivity_uncertainty),
            decoding_instability_uncertainty=clamp_probability(decoding_instability_uncertainty),
            epistemic_risk=epistemic_risk,
            raw=raw or {},
        )
