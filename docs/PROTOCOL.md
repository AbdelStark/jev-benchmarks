# BTZSC pilot v1 protocol

Status: frozen before any pilot model inference.

## Scientific question

On public, ground-truth, zero-shot text-classification tasks, how do Jev and
`fastino/gliner2.5-multi-v1` compare in discrimination, probability quality, selective automation,
and end-to-end latency when they receive the same text and semantic label descriptions?

This is a shared-capability comparison, not a claim about either system's overall intelligence.

## Hypotheses and success criteria

The primary outcome is macro-F1, averaged equally across datasets. Probability quality (Brier,
negative log likelihood, ECE), coverage at a 5% empirical error budget, and latency are co-primary
engineering outcomes. We do not declare a universal winner. A difference is considered supported in
this pilot only when its paired 95% bootstrap interval excludes zero on the evaluated slice.

Falsifying outcomes include Jev failing to expose normalized probabilities, either backend producing
schema-invalid answers, or rankings reversing substantially by dataset. Failures remain in the
denominator and are reported separately; they are never silently retried into disappearance.

## Data

BTZSC revision `fef2a2ac62b69c58670047dddf045c53d7c3cb5e` provides canonical zero-shot
label descriptions. The pilot uses 100 deterministically selected test examples from each of:

| Dataset | Family | Purpose |
| --- | --- | --- |
| AG News | topic, 4 labels | low-cardinality topical routing |
| DAIR Emotion | emotion, 6 labels | semantically adjacent affect labels |
| Banking77 | intent, 72 labels in BTZSC | high-cardinality fine-grained routing |

Sampling is deterministic and approximately class-balanced. The manifest stores the selected example
IDs, text hashes, labels, and source revision. No training examples or task-specific demonstrations
are supplied. Because public checkpoints may have seen these datasets, results measure benchmark
performance, not uncontaminated generalization.

The pinned BTZSC Banking77 configuration exposes 72 candidate hypotheses and includes 200
out-of-scope rows with no positive candidate. Those rows are excluded rather than adopting the
upstream loader's fallback that assigns no-positive examples to class 0. The exclusion is determined
solely from ground-truth structure before either model is run.

## Model inputs

Both backends receive the exact input text and ordered label descriptions. Jev gets one `Choice`
question whose stable option IDs map to those descriptions. GLiNER2.5 gets one exclusive
classification schema. Label order is fixed by BTZSC. There is no per-dataset prompt tuning.

GLiNER2.5 is pinned to model revision `235cf92d6d4318da9bfca0d08975c8fa7250d13b`, uses CPU,
and is timed at batch size one for an online-decision comparison. Model loading and the first warm-up
call are excluded from per-example latency. Jev uses `jev-latest`; the response's resolved model name
is recorded on every example because the alias may move.

## Metrics

For each dataset and model: accuracy, macro-F1, multiclass Brier score, negative log likelihood,
top-label ECE (10 equal-width bins), p50/p95 latency, failures, and coverage at a 5% empirical error
budget. Coverage is selected on the same pilot slice and is therefore descriptive, not a deployable
threshold estimate. A future confirmatory run must choose thresholds on validation data and evaluate
them once on a disjoint test set.

Paired, target-stratified bootstrap resampling uses 2,000 deterministic resamples so the deliberately
balanced class composition is preserved. Dataset macro-averages weight each dataset equally. Raw
example-level prediction vectors are retained locally.

## Controls and limitations

- Negative control: a uniform probability predictor is included in reporting.
- Integrity control: every vector must contain exactly the declared labels, finite values in `[0,1]`,
  and sum to one within tolerance.
- Failure control: failed examples are explicit and prevent paired comparison for that item.
- Cost: Jev token usage is recorded when the API reports it. GLiNER hardware and process metadata are
  recorded; this pilot does not convert local compute to a dollar estimate.
- The sample is too small for stable conclusions on rare Banking77 classes or production thresholds.
- Hosted-network latency and local CPU latency measure different deployment modes and must be labeled
  as such.
