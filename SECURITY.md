# Security

## Sensitive data

Prompts, answers, references, subgroup metadata, and model responses may contain personal, confidential, regulated, or proprietary information. Structured logs apply configured redaction patterns. Audit artifacts do not redact source data because complete reconstruction is required. Protect the artifact root and database as sensitive data stores.

## Authentication

The HTTP API accepts keys through `X-API-Key`. Keys are compared against the configured set in `EUQ_API_KEYS`. Deploy behind TLS. Use random keys with sufficient entropy and rotate them through overlapping key sets.

## Rate limiting

Redis provides a distributed fixed-window limit per API key. When Redis is unavailable, each process uses an in-memory rolling window. Multi-replica deployments must monitor Redis readiness because local fallback permits a higher aggregate rate.

## Filesystem access

Batch evaluation resolves dataset and optional configuration paths and requires them to be descendants of `EUQ_ALLOWED_DATA_ROOT`. Symlink resolution occurs before the containment check. The subprocess backend command is administrator configuration and must never be writable by untrusted API clients.

## Backend credentials

HTTP backend credentials are loaded from the environment variable named by `api_key_env`. Credentials are never included in the serialized backend configuration or request reproducibility metadata. Additional headers configured directly in backend files are serialized and therefore must not contain secrets.

## Dependency and image management

Build images from reviewed dependency manifests. Scan Python dependencies and container images. Pin production image digests in deployment infrastructure. Rebuild after security updates and re-run the full test and calibration evaluation suite.

## Reporting vulnerabilities

Report vulnerabilities privately to the deployment owner with affected version, configuration, reproduction steps, and impact. Do not include live credentials, private prompts, or unredacted user data in a report.
