# Changelog

All notable user-visible changes are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Apache-2.0 license, citation metadata, contribution and security policies.
- GitHub Actions gates for tests, coverage, lint, formatting, typing, package builds, and wheel smoke.
- Machine-readable aggregate artifact for the BTZSC pilot.
- Architecture and artifact-contract documentation.

### Changed

- Heavy data, GLiNER, and Jev runtimes are optional extras with lazy imports.
- Reports verify exact manifest coverage, targets, ordered labels, model identities, and artifact hashes.
- Selective coverage now evaluates only realizable thresholds and never splits confidence ties.
- Test coverage expanded to adapter, CLI, config, data, I/O, report, metric, and runner contracts.

## [0.1.0] - 2026-09-17

### Added

- Reproducible BTZSC pilot harness for Jev and GLiNER2.5.
- Frozen protocol, pinned configs, resumable JSONL predictions, calibration metrics, and paired
  bootstrap intervals.
- First 300-example aggregate result across AG News, DAIR Emotion, and Banking77/BTZSC.

[Unreleased]: https://github.com/AbdelStark/jev-benchmarks/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AbdelStark/jev-benchmarks/releases/tag/v0.1.0
