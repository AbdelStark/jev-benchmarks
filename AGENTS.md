# Repository guidance

## Research integrity

- Treat performance claims as hypotheses until they are backed by versioned artifacts.
- Freeze experiment configs before model inference. Never tune on the reported test slice.
- Preserve per-example predictions, timings, resolved model revisions, usage, and failures.
- Compare models only on shared capabilities with identical examples, labels, and scoring rules.
- Label pilots as pilots and state contamination, sampling, latency, and threshold limitations.
- Never commit API keys, `.env` files, downloaded checkpoints, caches, or raw licensed datasets.

## Development contract

```bash
uv sync --extra benchmark --dev
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

Keep heavyweight runtimes behind optional extras and imports lazy. New backends require deterministic
contract tests with fake boundaries; live credentials and network access are not CI dependencies.
Aggregate reports may be committed under `results/reports/`; raw runs stay under ignored
`results/runs/`.
