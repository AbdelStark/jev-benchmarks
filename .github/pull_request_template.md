## Summary

Describe the benchmark or software contract changed by this pull request.

## Evidence

- [ ] Tests cover the new behavior or failure mode.
- [ ] Dataset/model/config revisions are pinned where relevant.
- [ ] Reported claims link to reproducible artifacts.
- [ ] No credentials, raw sensitive inputs, caches, or model weights are included.

## Validation

```text
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

## Limitations

State what this change does not establish or validate.
