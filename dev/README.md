# Internal development utilities

Not part of the public workflow and not included in the installed package.
Users should run the root `train.py` and `test.py` commands.

This directory retains optional intermediate commands, input audits, alternate
metric analyses, and native-source compatibility checks. They exist for
maintainers, not as alternative instructions for evaluating the paper model.
Run them from a checkout with the package installed. Private data and weights
are required for server checks; nothing is downloaded automatically.

`commands/` contains step-level wrappers. `evaluate_mesh.py` is supplementary
area/vertex analysis, not the default evaluator. Store run manifests, logs,
and verification records outside the public repository. No audit utility
belongs to the DentalPSAM API.

Run `bash tests/run_checks.sh` and `pytest -q` for regression checks, and
`python dev/check_git_payload.py` before committing. Maintain `main` only.
