# Epistemic UQ

Epistemic UQ is a complete uncertainty-quantification, confidence-calibration, selective-prediction, and audit system for language-model outputs. Its objective is to estimate the probability that a produced answer is correct, separate model-knowledge uncertainty from prompt sensitivity and decoding instability, calibrate confidence against held-out correctness labels, and convert calibrated risk into explicit downstream actions.

The system supports black-box OpenAI-compatible HTTP APIs, Ollama, local Hugging Face causal language models, and any local executable that implements the included JSON protocol. It preserves prompts, requests, generations, token probabilities, confidence queries, truth-verification responses, stochastic samples, prompt perturbations, semantic clusters, features, labels, policy decisions, costs, latency, and reproducibility metadata in content-addressed audit artifacts.

## Implemented uncertainty signals

The baseline answer is evaluated with the following independent signals when the selected backend supports them:

1. Numeric self-reported confidence obtained through a constrained JSON confidence query.
2. Answer-level confidence aggregated from token log probabilities with geometric mean, arithmetic mean, minimum, product, or length-normalized product.
3. Truth-verification probability from a constrained correct-versus-incorrect query.
4. Self-consistency from repeated stochastic generations under a fixed prompt.
5. Prompt-perturbation stability under deterministic instruction-order, formatting, task-framing, and lexical transformations.
6. Cross-model agreement over normalized baseline answers.
7. Semantic agreement, dominant cluster mass, semantic entropy, lexical agreement, and contradiction detection.
8. Composite epistemic risk decomposed into model-knowledge uncertainty, prompt-sensitivity uncertainty, and decoding-instability uncertainty.
9. Transparent learned fusion or deterministic rule-based fusion.
10. Post-hoc calibration with temperature scaling, Platt scaling, isotonic regression, or beta calibration.

## Architecture

The source tree is divided into explicit layers:

- `schemas.py` contains immutable Pydantic schemas for prompts, requests, generations, token probabilities, samples, answers, clusters, perturbations, calibration bins, labels, subgroup audits, decisions, manifests, drift snapshots, and experiment runs.
- `backends` contains production adapters for OpenAI-compatible HTTP, Ollama, Hugging Face, and subprocess JSON protocol backends.
- `processing` contains dataset normalization, answer canonicalization, deterministic perturbation generation, and exact, regex, numeric, token-F1, and structured validators.
- `uncertainty` contains self-report parsing, log-probability aggregation, semantic adjudication, clustering, agreement statistics, contradiction detection, and epistemic risk decomposition.
- `calibration` contains reliability metrics, selective prediction metrics, four supervised calibrators, transparent monotonic fusion, deterministic fusion, subgroup audits, grouping-loss proxies, and worst-slice discovery.
- `policy.py` converts calibrated confidence and risk adjustments into answer, warning, clarification, verification, abstention, or human-escalation actions.
- `storage.py` contains relational experiment metadata and content-addressed compressed artifact persistence.
- `runner.py` executes complete experiments, parallelizes examples, resumes completed work, preserves every intermediate artifact, and records drift observations.
- `reports.py` emits machine-readable JSON, human-readable HTML, reliability diagrams, confidence histograms, coverage-versus-accuracy curves, worst-slice tables, contradiction cases, and overconfident errors.
- `service.py` exposes authenticated, rate-limited HTTP APIs, Prometheus metrics, health checks, structured logs, and trace identifiers.
- `cli.py` exposes dataset, backend, experiment, metrics, report, threshold, calibration, and fusion commands.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e . --no-deps
cp .env.example .env
alembic upgrade head
```

For Hugging Face and embedding-based semantic equivalence:

```bash
python -m pip install torch transformers sentence-transformers
```

For tests:

```bash
python -m pip install -r requirements-test.txt
python -m pip install --no-build-isolation -e . --no-deps
pytest
```

## Immediate end-to-end validation

The repository includes a deterministic local reference engine that implements the same subprocess protocol as a local model process. It performs real arithmetic, comparison, sentiment, extraction, and structured-output tasks and returns aligned token probabilities. The end-to-end command runs baseline generation, self-report, truth verification, stochastic sampling, prompt perturbations, evaluation, persistence, calibration metrics, policy inference, and report export.

```bash
cp .env.example .env
python scripts/bootstrap.py
python scripts/e2e.py
```

The experiment identifier is printed as JSON. Reports are written under `var/reports/<experiment_id>/report.html` and `var/reports/<experiment_id>/report.json`.

## Backend registration

Register the local reference backend:

```bash
epistemic-uq backend register --path examples/backend-subprocess.json
```

Register Ollama after the model is installed locally:

```bash
ollama pull llama3.1:8b
epistemic-uq backend register --path examples/backend-ollama.json
```

Register a Hugging Face model:

```bash
epistemic-uq backend register --path examples/backend-huggingface.json
```

Register an OpenAI-compatible server running at `http://localhost:8080/v1/chat/completions`:

```bash
epistemic-uq backend register --path examples/backend-http.json
```

The HTTP adapter sends chat-completion requests with temperature, top-p, maximum tokens, optional seed, and optional log-probability fields. Provider-specific headers and payload fields can be supplied through the backend configuration `options.headers` and `options.payload` mappings. Secret values are loaded from the environment when `api_key_env` is configured.

## Dataset format

JSON Lines, JSON arrays, JSON objects containing an `examples` array, and CSV are accepted. Every normalized record has the following fields:

```json
{
  "example_id": "math-001",
  "dataset_id": "reference-benchmark",
  "task_type": "question_answering",
  "user_input": "What is 17 + 25?",
  "expected_format": "number",
  "reference_label": 42,
  "valid_answers": [],
  "subgroup_metadata": {
    "domain": "arithmetic",
    "difficulty": "easy"
  },
  "perturbation_rules": {},
  "validator_config": {
    "method": "numeric"
  },
  "criticality": "low",
  "metadata": {}
}
```

Valid task types are `question_answering`, `classification`, `extraction`, and `structured`. Valid validator methods are `auto`, `regex`, `numeric`, `structured`, and `token_f1`. Numeric validators accept `absolute_tolerance` and `relative_tolerance`. Structured validators accept `required_keys`. Token-F1 validators accept `threshold`.

## Experiment execution

```bash
epistemic-uq dataset register \
  --path examples/dataset.jsonl \
  --dataset-id reference-benchmark \
  --version 1

epistemic-uq experiment run \
  --dataset examples/dataset.jsonl \
  --dataset-id reference-benchmark \
  --backend reference-local \
  --config config/e2e.yaml
```

Multiple `--backend` options enable cross-model agreement:

```bash
epistemic-uq experiment run \
  --dataset data/evaluation.jsonl \
  --dataset-id evaluation-v1 \
  --backend ollama-local \
  --backend smollm2-local \
  --config config/default.yaml
```

Interrupted experiments resume by identifier without re-running completed example-backend pairs:

```bash
epistemic-uq experiment run \
  --dataset data/evaluation.jsonl \
  --dataset-id evaluation-v1 \
  --backend ollama-local \
  --resume "$EXPERIMENT_ID"
```

Resume validation rejects changed dataset hashes or changed configuration hashes.

## Metrics and reports

```bash
epistemic-uq metrics compute \
  --experiment-id "$EXPERIMENT_ID" \
  --source calibrated_confidence \
  --bins 15 \
  --strategy quantile

epistemic-uq report generate \
  --experiment-id "$EXPERIMENT_ID" \
  --output-dir "var/reports/$EXPERIMENT_ID"
```

Metrics include accuracy, reliability bins, expected calibration error, maximum calibration error, Brier score, negative log likelihood, AUROC when both classes are present, area under the risk-coverage curve, coverage, accuracy, risk, and abstention counts.

## Supervised calibration

Calibration is fit only on a development experiment and evaluated on a distinct held-out test experiment:

```bash
epistemic-uq calibration fit \
  --development-experiment "$DEVELOPMENT_EXPERIMENT_ID" \
  --test-experiment "$TEST_EXPERIMENT_ID" \
  --source self_consistency_confidence \
  --method isotonic \
  --calibrator-id self-consistency-isotonic-v1
```

The stored artifact contains fitted parameters, development metrics before and after calibration, test metrics before and after calibration, source identity, experiment identities, fitting timestamp, and a training-data hash. Add the calibrator to a pipeline configuration with:

```yaml
calibration:
  calibrator_id: self-consistency-isotonic-v1
```

## Transparent confidence fusion

Fusion uses strict train, validation, and test experiment separation. The validation experiment selects the L2 regularization strength. The final transparent model is fitted on train plus validation only after model selection, and the test experiment remains untouched until final evaluation.

```bash
epistemic-uq fusion fit \
  --train-experiment "$TRAIN_EXPERIMENT_ID" \
  --validation-experiment "$VALIDATION_EXPERIMENT_ID" \
  --test-experiment "$TEST_EXPERIMENT_ID" \
  --model-id transparent-fusion-v1 \
  --monotonic
```

Add the model to configuration with:

```yaml
fusion:
  mode: learned
  model_id: transparent-fusion-v1
```

The model exports feature names, nonnegative monotonic coefficients, intercept, normalization statistics, missing-value imputation values, selected regularization, train metrics, validation metrics, test metrics, and training hashes.

## Threshold simulation

```bash
epistemic-uq threshold simulate \
  --experiment-id "$EXPERIMENT_ID" \
  --source calibrated_confidence \
  --objective minimum_risk \
  --max-risk 0.05
```

Supported objectives are `utility`, `minimum_risk`, and `target_coverage`. Utility optimization accepts correct, incorrect, and abstention utilities. Minimum-risk optimization returns the highest-coverage feasible threshold. Target-coverage optimization returns the operating point closest to the requested coverage.

## Service API

Start the service:

```bash
uvicorn epistemic_uq.service:app --host 0.0.0.0 --port 8000
```

The default local API key from `.env.example` is `local-development-key`.

Health and metrics:

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/metrics
```

Single-query uncertainty estimation:

```bash
curl -sS http://localhost:8000/v1/uncertainty \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: local-development-key' \
  -d '{
    "backend_ids": ["reference-local"],
    "task": {
      "example_id": "api-math-1",
      "dataset_id": "api",
      "task_type": "question_answering",
      "user_input": "What is 21 + 21?",
      "expected_format": "number",
      "reference_label": 42,
      "valid_answers": [],
      "subgroup_metadata": {"domain": "arithmetic"},
      "perturbation_rules": {},
      "validator_config": {"method": "numeric"},
      "criticality": "low",
      "metadata": {}
    },
    "generation": {},
    "config_overrides": {}
  }'
```

The response contains the normalized answer, full baseline generation, raw uncertainty signals, epistemic decomposition, calibrated confidence, evaluation label when a reference exists, policy action, reasons, and audit artifact path.

Policy inference:

```bash
curl -sS http://localhost:8000/v1/policy/decide \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: local-development-key' \
  -d '{
    "calibrated_confidence": 0.72,
    "features": {
      "contradiction": false,
      "model_knowledge_uncertainty": 0.2,
      "prompt_sensitivity_uncertainty": 0.1,
      "decoding_instability_uncertainty": 0.15,
      "epistemic_risk": 0.18,
      "raw": {}
    },
    "criticality": "medium",
    "subgroup_metadata": {},
    "subgroup_audits": []
  }'
```

Threshold simulation:

```bash
curl -sS http://localhost:8000/v1/thresholds/simulate \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: local-development-key' \
  -d '{
    "confidences": [0.95, 0.8, 0.45, 0.2],
    "labels": [1, 1, 0, 0],
    "utilities": {"objective": "minimum_risk", "max_risk": 0.05}
  }'
```

Batch evaluation accepts server-local dataset and configuration paths only when both resolve inside `EUQ_ALLOWED_DATA_ROOT`. This prevents arbitrary filesystem reads.

## Subprocess backend protocol

The adapter starts the configured command once per request, writes one JSON object to standard input, and reads one JSON object from standard output. The input includes request identifier, model, prompt, temperature, top-p, maximum tokens, seed, stop strings, log-probability request, and metadata. The output requires `text` and supports `model`, `finish_reason`, `token_probabilities`, `usage`, and `reproducibility`.

Token-probability records contain `token`, `logprob`, optional `probability`, `position`, optional `start_char`, and optional `end_char`. Character spans allow answer-specific token selection for multi-part outputs.

## Persistence and auditability

Relational metadata is stored through SQLAlchemy in SQLite or PostgreSQL. Raw and derived artifacts are serialized as canonical sorted JSON, compressed with deterministic gzip metadata, hashed with SHA-256, written atomically, and addressed by content hash. Every result links to an audit artifact containing:

- Baseline request and generation.
- Self-report generation and parsed confidence.
- Truth-verification generation and parsed probability.
- Every stochastic sample and normalized answer.
- Every semantic cluster and agreement statistic.
- Every perturbation definition, request, generation, answer, and equivalence judgment.
- Cross-model baseline answers and generation identifiers.
- Uncertainty features and decomposition.
- Fusion contributions and post-calibration probability.
- Evaluation label and policy decision.

Experiment manifests preserve dataset hashes, backend configurations, model identifiers, prompt template identifiers and versions, decoding parameters, seeds, timestamps, configuration, source paths, and backend reproducibility metadata.

## Observability

Structured JSON logs include trace identifiers. Every HTTP response contains `X-Trace-ID` and `X-Response-Time-Ms`. Prometheus metrics cover model calls, backend errors, latency, prompt tokens, completion tokens, parsing failures, policy actions, recent confidence means, and drift alarms.

The drift monitor computes population stability index over reference and current windows and optionally compares confidence-versus-correctness error. Drift observations are segmented by signal, task type, and subgroup metadata and are persisted when a complete window is available.

Configured redaction removes email addresses, telephone numbers, credit-card-like sequences, and optional IP addresses from structured logs. Raw audit artifacts remain unredacted because they are the source record and must be protected with filesystem, database, encryption, retention, and access-control policies appropriate to the deployment.

## Deployment

Local containers:

```bash
docker compose up --build
```

The Compose stack runs the API, PostgreSQL 16, and Redis 7. PostgreSQL stores experiment metadata. Redis provides distributed fixed-window API rate limiting. Artifact data is stored in a persistent volume.

Before a network deployment:

1. Replace the development API key with at least one cryptographically random key in `EUQ_API_KEYS`.
2. Terminate TLS at a trusted reverse proxy or service mesh.
3. Restrict artifact and database access to the API identity.
4. Configure PostgreSQL backups and artifact-volume snapshots.
5. Set retention requirements for prompts and generations.
6. Configure log shipping and Prometheus scraping.
7. Run migrations with `alembic upgrade head` before deploying the new application version.
8. Validate backend capability flags against the selected provider.
9. Fit calibrators only from labeled development data collected under the same task and backend conditions.
10. Re-audit subgroup calibration after model, prompt, dataset, or policy changes.

## Test coverage

The test suite covers immutable schema validation, backend configuration validation, text and structured normalization, numeric tolerance, confidence parsing, truth parsing, token-probability aggregation, semantic equivalence, semantic clustering, contradiction detection, calibration metrics, selective curves, calibrator serialization, monotonic fusion, rule fusion, subgroup audits, worst-slice discovery, policy actions, artifact persistence, relational persistence, subprocess protocol execution, redaction, drift detection, complete single-example execution, complete dataset execution, resume behavior, and report generation. Hypothesis tests validate selective-curve probability and risk invariants across generated confidence arrays.
