# Contributing

Keep changes focused and explain their effect on training, inference, or evaluation.
The maintained branch is `main`; preserve other contributors' work and avoid
rewriting published history.

## Checks

Run the unit tests and regression suite in the documented environment:

```bash
pytest -q
bash tests/run_checks.sh
git diff --check
```

Add behavioral tests when changing labels, normalization, checkpoint loading,
face ordering, padding, fusion, or metrics. The default evaluation uses equal
triangle weights, strict `> 0.5` decisions, fixed 0.5/0.5 fusion, and per-mesh
averaging. Keep participants separate across train, validation, and test.

## Repository contents

Commit source code, tests, documentation, paper figures, and the published
example cases under `examples/`. Keep full datasets, additional predictions,
run logs, generated JSON reports, and private experiment records outside Git.
Publish large pretrained checkpoints as GitHub Release assets.
Tests should create their temporary inputs at runtime.
Never commit credentials or private storage paths.

Use concise commit messages that describe the change. Review staged files and
push tested, coherent updates to `main`.
