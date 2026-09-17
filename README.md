# jev-benchmarks

Private, reproducible evaluation infrastructure for TypeSafe's Jev and comparable decision models.
The first comparison is `fastino/gliner2.5-multi-v1` on the shared capability both systems expose:
zero-shot, single-label text classification with per-label probabilities.

This project deliberately does **not** compare Jev with GLiNER2.5 on named-entity recognition,
relation extraction, or record extraction because Jev does not expose equivalent outputs. The pilot
uses fixed slices of the public BTZSC benchmark and scores both systems from preserved probability
vectors.

## What the benchmark reports

- accuracy and macro-F1;
- multiclass Brier score and negative log likelihood;
- top-label expected calibration error (ECE);
- risk/coverage and automation coverage at a fixed empirical error budget;
- end-to-end latency, input tokens when reported, failures, and model/dataset revisions;
- paired bootstrap confidence intervals for model differences.

See [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the frozen pilot design and limitations.

## Setup

```bash
uv sync --all-extras --dev
```

Jev reads `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, and `TYPESAFE_DEFAULT_MODEL` from the environment.
Keep them outside this repository. GLiNER downloads its pinned Hugging Face revision on first use.

## Run

```bash
# Validate config and materialize the exact evaluation examples without model calls.
uv run jev-bench prepare --config configs/pilot-v1.yaml

# One backend at a time; runs resume safely from existing JSONL predictions.
uv run jev-bench run --config configs/pilot-v1.yaml --backend gliner
uv run jev-bench run --config configs/pilot-v1.yaml --backend jev

# Produce machine-readable and Markdown summaries from raw predictions.
uv run jev-bench report --config configs/pilot-v1.yaml
```

Run artifacts live under `results/runs/<experiment-id>/` and are ignored by git. Curated summaries
may be copied to `results/reports/` after checking that they contain no sensitive state.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

No result from this repository is approved for public release. TypeSafe receives any evaluation
privately first, and only after explicit owner approval.
