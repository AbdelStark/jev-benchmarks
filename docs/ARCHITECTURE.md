# Architecture

`jev-benchmarks` separates experiment definition, data selection, inference, validation, and analysis
so a model call can never silently rewrite the evaluation contract.

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Load and validate the YAML experiment contract; resolve artifact paths. |
| `data.py` | Load a pinned BTZSC revision, reject targetless rows, sample deterministically, write the manifest. |
| `adapters/` | Translate a common `Example` into backend-specific calls and return `Prediction`. |
| `runner.py` | Resume append-only runs, validate probability vectors, retain failures and usage. |
| `metrics.py` | Compute discrimination, calibration, selective-risk, latency, controls, and paired intervals. |
| `report.py` | Verify manifest/prediction identity and produce hashed JSON and Markdown summaries. |
| `io.py` | Atomic JSONL writes, durable appends, artifact hashing, and runtime metadata. |

## Artifact flow

### Manifest

`prepare` writes one JSON object per selected example:

```json
{
  "dataset": "agnews",
  "task": "topic",
  "example_id": "agnews:9",
  "text": "...",
  "text_sha256": "...",
  "labels": ["..."],
  "target_index": 0
}
```

The ordered `labels` tuple is part of the evaluation contract. If an existing manifest differs from
the deterministic regeneration, preparation fails and requires a new experiment directory.

### Predictions

Each backend writes an append-only JSONL record containing requested and resolved model identities,
the complete probability vector, target and predicted indices, latency, optional token usage, the raw
probability sum, and any error. Successful example IDs are skipped on resume. Failed calls remain in
the log and are retried; reporting selects the most recent record for each example.

### Report

Before scoring, report generation checks that every backend covers exactly the manifest IDs and that
each target and ordered label tuple matches the manifest. The JSON report includes:

- schema and package versions;
- the complete experiment config and protocol revision;
- config, manifest, and prediction-log SHA-256 hashes;
- resolved model identities;
- runtime metadata;
- per-model/per-dataset metrics;
- paired confidence intervals.

## Probability validation

Vectors must have one finite value in `[0, 1]` for every label. Sums within `0.02` of one are treated
as serialization rounding and normalized; the pre-normalization sum is retained. Larger deviations
are failures.

The tolerance addresses APIs that expose rounded probabilities. It is not a general repair for
unnormalized scores, logits, or independent one-vs-rest probabilities.

## Selective coverage

Coverage at an error budget is computed only at thresholds realizable from reported confidence
values. Every example tied at a threshold is accepted together. This avoids optimistic coverage that
could only be achieved by splitting identical scores.

The pilot chooses and evaluates the threshold on one slice. A confirmatory experiment should select
the threshold on validation data and report risk once on a disjoint test set.

## Extension points

Backends implement the `Backend` protocol in `adapters/base.py`. Dataset loaders emit the common
`Example` type. The current CLI lists supported backend names explicitly so adding a provider is a
reviewed API change rather than dynamic code loading.

Heavy dependencies are optional:

- `data`: Hugging Face datasets;
- `gliner`: GLiNER2, PyTorch, Transformers, tokenizer dependencies;
- `jev`: the TypeSafe SDK;
- `benchmark`: the complete stack.

Imports occur only when the corresponding command/backend is selected, so metrics and report tooling
remain lightweight.
