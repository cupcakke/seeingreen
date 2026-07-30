from epistemic_uq.calibration.subgroups import audit_subgroups, discover_worst_slices, grouping_loss_proxy
from epistemic_uq.policy import DecisionPolicy
from epistemic_uq.schemas import Criticality, DecisionAction, PolicyConfig, UncertaintyFeatures


def features(contradiction: bool = False) -> UncertaintyFeatures:
    return UncertaintyFeatures(
        contradiction=contradiction,
        model_knowledge_uncertainty=0.2,
        prompt_sensitivity_uncertainty=0.1,
        decoding_instability_uncertainty=0.1,
        epistemic_risk=0.15,
    )


def test_grouping_loss_nonnegative() -> None:
    assert grouping_loss_proxy([0.1, 0.2, 0.8, 0.9], [0, 1, 0, 1], n_bins=2) >= 0.0


def test_subgroup_audit_finds_overconfidence() -> None:
    confidences = [0.9] * 40 + [0.6] * 40
    labels = [0] * 40 + [1] * 40
    metadata = [{"group": "bad"}] * 40 + [{"group": "good"}] * 40
    audits = audit_subgroups(confidences, labels, metadata, minimum_size=20)
    assert audits[0].subgroup_value == "bad"
    worst = discover_worst_slices(confidences, labels, metadata, minimum_size=20)
    assert worst[0]["slice"]["group"] == "bad"


def test_policy_escalates_contradiction() -> None:
    policy = DecisionPolicy(PolicyConfig())
    decision = policy.decide(0.95, features(True), Criticality.LOW)
    assert decision.action == DecisionAction.ESCALATE


def test_policy_answers_high_confidence() -> None:
    policy = DecisionPolicy(PolicyConfig(criticality_adjustments={"low": 0.0}))
    decision = policy.decide(0.95, features(False), Criticality.LOW)
    assert decision.action == DecisionAction.ANSWER
