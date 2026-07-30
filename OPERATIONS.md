# Operations

## Database migration

```bash
alembic upgrade head
```

Downgrade the initial schema only when all stored data may be removed:

```bash
alembic downgrade base
```

## Readiness

`/health/live` verifies process availability. `/health/ready` checks database connectivity and reports Redis connectivity. Database failure marks readiness unavailable. Redis failure is reported separately because the service retains a process-local rate limiter.

## Backup

Back up PostgreSQL and the artifact root at a mutually consistent point. Database result records contain artifact paths, so both stores are needed for complete recovery. SQLite deployments must stop writers or use SQLite online backup before copying the database file.

## Restore validation

After restore:

```bash
alembic upgrade head
epistemic-uq backend list
curl http://localhost:8000/health/ready
```

Select an experiment and verify that every completed result artifact path exists and is readable. Regenerate its report and compare the report JSON hash with the pre-backup hash when available.

## Scaling

Use PostgreSQL rather than SQLite for concurrent workers. Put Redis behind authentication and network policy. Run multiple API replicas behind a load balancer. Artifact storage must be shared, durable, and support atomic rename semantics. For object storage, mount through a consistency layer or implement an artifact-store adapter with conditional writes and content hashes.

## Capacity

Storage grows with the number of model calls rather than only the number of examples. Per example and backend, the call count is one baseline, one optional self-report, one optional truth verification, the configured sample count, and one call per perturbation. Cross-model execution multiplies this by backend count. Token traces and raw provider responses can dominate artifact size.

## Incident response

For backend error spikes, inspect `euq_model_calls_total` by backend and status, then inspect structured logs by trace identifier. For latency changes, inspect `euq_model_call_latency_seconds`. For calibration alarms, inspect stored drift snapshots and regenerate subgroup reports. For parsing failures, inspect `euq_parsing_failures_total` and the corresponding audit generation text.

## Key rotation

Set `EUQ_API_KEYS` to a comma-separated set containing both old and new keys, deploy, migrate clients, remove the old key, and deploy again. Never write production keys into repository files.
