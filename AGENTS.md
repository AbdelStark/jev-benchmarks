# Jev benchmarks

This repository is private. It evaluates TypeSafe's Jev and other decision models.

## Research integrity

- Treat TypeSafe performance claims as hypotheses, not facts.
- Freeze experiment configs before live model calls. Never tune on the reported test slice.
- Preserve raw per-example predictions, timings, model revisions, dataset revisions, and failures.
- Compare models only on shared capabilities and identical examples, labels, and scoring rules.
- Label pilot results as pilots; do not generalize beyond the evaluated datasets and sample sizes.
- Do not publish or send TypeSafe evaluations without explicit in-session owner approval.
- Never commit API keys, `.env` files, downloaded model weights, or Hugging Face caches.

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check .
uv run pyright
```

Run benchmark commands from this repository. Results under `results/` may contain private model
outputs; only explicitly selected, redacted result summaries belong in git.
