# Contributing

Contributions that improve benchmark correctness, reproducibility, model coverage, or analysis are
welcome.

## Before opening a pull request

For a new dataset, metric, or backend, open an issue first. Include:

- the scientific question;
- why the compared output contracts are equivalent;
- dataset license and contamination considerations;
- the primary metric and failure policy;
- expected runtime, hardware, API, and monetary cost.

Bug fixes and documentation corrections can go directly to a pull request.

## Development setup

```bash
git clone https://github.com/AbdelStark/jev-benchmarks.git
cd jev-benchmarks
uv sync --extra benchmark --dev
```

The core package can be developed without heavyweight extras using `uv sync --dev`.

## Required checks

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

Tests must not require credentials, network access, model downloads, or an accelerator. Use fake
provider/model boundaries for contract tests and mark any separately maintained live smoke clearly.

## Research changes

Do not revise a frozen config in place after inspecting model results. Create a new experiment ID,
state what changed, and preserve the earlier artifact. Report all evaluated conditions and failures;
do not select datasets or metrics based on favorable outcomes.

New reported results should include:

- exact reproduction commands;
- pinned code, dataset, and checkpoint identities;
- sample counts and selection policy;
- raw-artifact hashes;
- uncertainty where comparisons are made;
- limitations and falsifying evidence;
- a machine-readable aggregate artifact.

Never commit credentials, raw restricted data, prediction logs containing sensitive input text,
downloaded checkpoints, or caches.

## Pull requests

Keep changes focused and explain the contract they alter. Add a changelog entry for user-visible
behavior. By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
