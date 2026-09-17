# BTZSC pilot v1: Jev vs GLiNER2.5

Private evaluation. Not approved for public release or outreach.

Run date: 17 September 2026. Protocol frozen at tag `pilot-v1-preregistered` before model
inference. Jev resolved to `jev-1.13.0`; GLiNER used
`fastino/gliner2.5-multi-v1@235cf92d6d4318da9bfca0d08975c8fa7250d13b` locally on an Apple
M4 Max CPU with 12 PyTorch threads.

## Result

Jev clearly outperformed GLiNER2.5 on the topic and high-cardinality intent slices. On emotion,
accuracy was statistically indistinguishable while Jev's probability quality was substantially
worse. The pilot therefore supports a task-dependent conclusion, not a universal winner.

| Dataset | Model | Accuracy | Macro-F1 | Brier ↓ | NLL ↓ | ECE ↓ | Coverage at ≤5% empirical error | p50 latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AG News | GLiNER2.5 | 0.700 | 0.659 | 0.413 | 0.742 | 0.124 | 0.240 | 44.9 ms local CPU |
| AG News | Jev | **0.910** | **0.905** | **0.146** | **0.495** | **0.064** | **0.830** | 255.9 ms hosted |
| Banking77/BTZSC | GLiNER2.5 | 0.610 | 0.569 | 0.521 | 1.444 | 0.062 | 0.270 | 295.5 ms local CPU |
| Banking77/BTZSC | Jev | **0.870** | **0.857** | **0.179** | **1.064** | **0.054** | **0.870** | **246.4 ms hosted** |
| DAIR Emotion | GLiNER2.5 | 0.440 | 0.407 | **0.668** | **1.381** | **0.117** | **0.020** | **43.3 ms local CPU** |
| DAIR Emotion | Jev | 0.480 | 0.479 | 0.846 | 5.588 | 0.351 | 0.000 | 236.3 ms hosted |

Each row contains 100 deterministically sampled, approximately class-balanced test examples. There
were no final inference failures.

## Paired uncertainty

Intervals are 95% target-stratified paired bootstrap intervals with 2,000 resamples. Differences are
Jev minus GLiNER2.5; positive favors Jev for accuracy/F1, while negative favors Jev for Brier/NLL.

| Dataset | Metric | Difference | 95% CI |
| --- | --- | ---: | ---: |
| AG News | Accuracy | 0.210 | [0.130, 0.290] |
| AG News | Macro-F1 | 0.246 | [0.151, 0.345] |
| AG News | Brier | -0.267 | [-0.363, -0.171] |
| AG News | NLL | -0.247 | [-0.623, 0.344] |
| Banking77/BTZSC | Accuracy | 0.260 | [0.220, 0.300] |
| Banking77/BTZSC | Macro-F1 | 0.288 | [0.252, 0.327] |
| Banking77/BTZSC | Brier | -0.342 | [-0.390, -0.297] |
| Banking77/BTZSC | NLL | -0.380 | [-0.947, 0.170] |
| DAIR Emotion | Accuracy | 0.040 | [-0.070, 0.150] |
| DAIR Emotion | Macro-F1 | 0.072 | [-0.041, 0.184] |
| DAIR Emotion | Brier | 0.178 | [0.041, 0.315] |
| DAIR Emotion | NLL | 4.206 | [2.677, 6.033] |

## Interpretation

- Jev's accuracy advantage is large and interval-supported on AG News and Banking77/BTZSC.
- The Banking result is especially relevant to Jev's typed high-cardinality routing interface:
  Jev was both more accurate and slightly faster end to end than batch-size-one local CPU GLiNER.
- On short emotion utterances, the accuracy difference is unresolved. Jev assigned exactly zero
  probability to the true label on 16% of examples, versus 0% for GLiNER, producing severe NLL and
  calibration penalties. Mean top-label confidence was 0.819 for Jev against 0.438 for GLiNER.
- Jev used 316,861 reported input tokens. At TypeSafe's stated $0.042 per million input tokens, the
  implied run cost is approximately $0.0133; this is an estimate, not a billing receipt.
- Latency is deployment-specific: Jev includes network service time from France, while GLiNER is
  local CPU inference. It is not a hardware-normalized throughput comparison.

## Integrity notes

The pinned BTZSC `banking77` configuration has 72 candidate hypotheses and 200 out-of-scope rows with
no positive candidate. Those rows were excluded before inference rather than silently assigning them
to class 0, as the upstream loader does. Two final Jev vectors summed to 0.99 due to output rounding;
the raw sums were retained and the vectors normalized for proper scoring. Public benchmark data may
have appeared in either model's training data, so these numbers are benchmark performance rather than
clean evidence of uncontaminated generalization.

The 5% error-budget coverage uses thresholds selected on the same pilot slice and is descriptive. A
confirmatory study must select thresholds on validation data, evaluate once on a disjoint held-out
test set, increase sample sizes, and include more BTZSC datasets.

## Local evidence hashes

Raw example text and per-example probabilities remain ignored locally under `results/runs/`.

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5` |
| GLiNER predictions | `4ea0a71ffe7e157e154b9b667c86dbc515bb676d000ea341ac3db4fbbdfb1ae8` |
| Jev predictions, including four superseded validation failures and retries | `fc2cffee4ffe9c1870beee21f40805b1209e1c3b4f5ae8183d2e4450a3ec2216` |
