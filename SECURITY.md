# Security policy

## Supported versions

Security fixes are applied to the latest commit on `main` until versioned releases begin.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Do not open a public issue for a
suspected vulnerability, leaked credential, unsafe model artifact, or dataset privacy problem.

Include the affected revision, reproduction steps, impact, and any proposed mitigation. Remove API
keys, tokens, personal data, and proprietary model inputs from reports and logs.

## Benchmark data and credentials

The repository does not require secrets for tests or report analysis. Live Jev runs read credentials
from environment variables. Raw manifests and prediction logs are ignored because they may contain
dataset text or model inputs; only aggregate reports should be committed.
