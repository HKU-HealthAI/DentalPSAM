# Development and evidence workflow

Keep code changes, experiment inputs, and scientific conclusions independently
traceable. Source organization does not by itself establish paper reproduction.

## Git workflow

1. Fetch and inspect `git status --short --branch` before making changes. Preserve
   other contributors' work. Use a topic branch for subsequent changes.
2. Make focused commits, for example `model: ...`, `eval: ...`, `docs: ...`.
   Do not amend published history, force-push, or reset other people's changes.
3. Run the relevant tests on the experiment server in the documented environment.
   Record the exact source revision or source hashes with test evidence.
4. Review `git diff --check` and the staged diff. Stage specific files or
   explicitly reviewed directories rather than the entire workspace.
5. Push coherent, tested milestones. Verify that the remote branch points to
   the intended commit. A public release tag requires the remaining license,
   checkpoint, data-access, and reproduction checks in the README.

Use repository-local Git identity and configuration. Credentials belong in
the process environment or credential manager, never in remote URLs, files,
commands, logs, or commits. Never commit passwords or personal access tokens.

## What belongs in Git

Commit source, synthetic tests, documentation, non-sensitive example configs,
and compact aggregate verification reports. Do not commit patient data, mesh
lists containing participant identifiers, checkpoints, saved predictions,
environments, credentials, or per-patient results. Store those in a separate
access-controlled experiment directory. `.gitignore` is a safeguard, not a
replacement for staged-content review.

## Results and changes to numerical behavior

The default MICCAI evaluator uses **equal triangle weights**, strict `> 0.5`
decisions, UV-centre truncation, fixed 0.5/0.5 fusion, and per-mesh averaging.
Area-weighted evaluation is a separately named analysis. Never substitute its
results or confidence intervals into the original paper's table.

Any change to labels, split membership, normalization, checkpoint selection,
fusion, sampling, or aggregation needs an explicit protocol note and regression
test. Never filter cases by model performance. Do not claim reproduction until
the fixed data, checkpoint, prediction artifacts, and evaluator jointly support
the claim.

Each experiment should record Git commit, clean/dirty status, command,
environment, checkpoint hashes, split hash, evaluator hash, and output hashes.
For an uncommitted test candidate, retain a source SHA-256 manifest and later
check it against the committed tree. Keep original data read-only.
