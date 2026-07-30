from __future__ import annotations

from epistemic_uq.schemas import AbstentionDecision, Criticality, DecisionAction, PolicyConfig, SubgroupAudit, UncertaintyFeatures
from epistemic_uq.utils import clamp_probability


class DecisionPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(
        self,
        calibrated_confidence: float,
        features: UncertaintyFeatures,
        criticality: Criticality,
        subgroup_audits: tuple[SubgroupAudit, ...] = (),
        subgroup_metadata: dict[str, str] | None = None,
    ) -> AbstentionDecision:
        metadata = subgroup_metadata or {}
        adjustment = float(self.config.criticality_adjustments.get(criticality.value, 0.0))
        matching = [
            audit
            for audit in subgroup_audits
            if metadata.get(audit.subgroup_key) == audit.subgroup_value
        ]
        risk_multiplier = max((audit.risk_multiplier for audit in matching), default=1.0)
        adjusted = clamp_probability(calibrated_confidence - adjustment)
        adjusted = clamp_probability(adjusted / risk_multiplier)
        reasons: list[str] = []
        if adjustment > 0.0:
            reasons.append(f"criticality_adjustment:{adjustment:.6f}")
        if risk_multiplier > 1.0:
            reasons.append(f"subgroup_risk_multiplier:{risk_multiplier:.6f}")
        if features.contradiction:
            reasons.append("high_confidence_contradiction")
            return AbstentionDecision(
                action=self.config.contradiction_action,
                calibrated_confidence=calibrated_confidence,
                adjusted_confidence=adjusted,
                threshold=self.config.answer_threshold,
                reasons=tuple(reasons),
                policy_version=self.config.policy_version,
            )
        if adjusted >= self.config.answer_threshold:
            action = DecisionAction.ANSWER
            threshold = self.config.answer_threshold
        elif adjusted >= self.config.warning_threshold:
            action = DecisionAction.WARN
            threshold = self.config.warning_threshold
            reasons.append("confidence_below_answer_threshold")
        elif adjusted >= self.config.clarification_threshold:
            action = DecisionAction.CLARIFY
            threshold = self.config.clarification_threshold
            reasons.append("confidence_requires_clarification")
        elif adjusted >= self.config.external_verification_threshold:
            action = DecisionAction.VERIFY
            threshold = self.config.external_verification_threshold
            reasons.append("confidence_requires_external_verification")
        elif criticality in {Criticality.HIGH, Criticality.CRITICAL}:
            action = DecisionAction.ESCALATE
            threshold = self.config.external_verification_threshold
            reasons.append("low_confidence_high_criticality")
        else:
            action = DecisionAction.ABSTAIN
            threshold = self.config.external_verification_threshold
            reasons.append("confidence_below_operating_threshold")
        return AbstentionDecision(
            action=action,
            calibrated_confidence=calibrated_confidence,
            adjusted_confidence=adjusted,
            threshold=threshold,
            reasons=tuple(reasons),
            policy_version=self.config.policy_version,
        )
