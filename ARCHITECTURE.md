# Architecture

## System objective

For each language-model answer, the system estimates a calibrated correctness probability and an epistemic-risk decomposition. The decomposition separates evidence associated with model knowledge, prompt sensitivity, and decoding instability. The output is designed for downstream policies that must decide whether to answer, warn, ask for clarification, request external verification, abstain, or escalate.

## Data flow

1. A dataset record is validated into an immutable canonical evaluation unit.
2. A baseline prompt is produced from the task and expected response format.
3. Every configured backend produces a baseline generation with log probabilities when supported.
4. Answers are canonicalized according to task type.
5. The backend is queried for numeric self-reported confidence and truth-verification probability.
6. Repeated stochastic samples are generated under a fixed prompt.
7. Deterministic meaning-preserving prompt perturbations are generated and evaluated.
8. Baseline answers from multiple backends are compared.
9. Answers are adjudicated for lexical, numeric, label, and optional embedding equivalence.
10. Semantic clusters, dominant mass, entropy, agreement, and contradictions are computed.
11. Signals are decomposed into knowledge, prompt, and decoding uncertainty.
12. Signals are fused by a transparent learned model or deterministic weighted rule.
13. A supervised calibrator optionally transforms the fused probability.
14. The decision policy applies confidence thresholds, task criticality, subgroup risk, and contradiction rules.
15. Raw and derived artifacts are serialized before the compact result is persisted.
16. Evaluation metrics, subgroup audits, selective curves, reports, and drift observations are computed from persisted results.

## Immutability

External and cross-layer data is represented with frozen Pydantic models and forbidden extra fields. Mutation is confined to backend clients, calibrator training objects, relational persistence records, and bounded monitoring windows. Serialized artifacts are canonicalized and content-addressed.

## Backend isolation

Every backend implements the same synchronous contract. The runner applies retries, timeouts delegated to the adapter, cost accounting, metrics, and reproducibility capture. Backend-specific payloads remain in raw responses and reproducibility metadata.

## Failure boundaries

A failed model call raises a backend error after bounded retry. A failed example is persisted with structured error type and message. Completed result records are never overwritten by later failure handling. Experiment status is `failed` when any example fails, while successful example artifacts remain available for inspection and resume.

## Calibration isolation

Calibrator fitting requires a development experiment and a held-out test experiment. Fusion fitting requires train, validation, and test experiments. Model selection uses validation only. Final test evaluation occurs after selection and fitting without test labels entering optimization.

## Storage model

The relational database stores dataset manifests, backend configurations, experiment state, result indexes, calibration artifacts, and drift snapshots. Large raw records are stored as deterministic compressed JSON files. Atomic replacement prevents partially written artifacts.

## Security model

The service requires API-key authentication and rate limits each key. Batch paths are restricted to a configured data root. Logs are structured and redacted. Raw artifacts are deliberately complete and require deployment-level access control, encryption, retention, and backup policy.
