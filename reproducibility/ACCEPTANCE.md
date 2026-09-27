# Repository Release Gate

This stage covers public readability, software usability, and preservation of
existing scientific behavior. It does **not** require reproducing paper metric
values. The separate [Scientific Reproduction Gate](archive/PAPER_RESULTS.md)
remains open and is not silently completed by a code cleanup or Git push.

## Reader-facing requirements

- [x] Dataset status: being organized, not public; contact the authors for
  access. No patient data are distributed.
- [x] [Dataset specification](../docs/DATASET.md): actual folders, inputs, annotations,
  matching filenames, preprocessing, and participant-disjoint splits.
- [x] Public workflow uses 2D branch, 3D branch, fusion, and DentalPSAM.
  Internal implementation names are explained once in
  [Model and checkpoints](MODEL_AND_CHECKPOINTS.md).
- [x] Testing is one command, `python test.py`, with documented prerequisites,
  fixed-list evaluation, output paths, and missing-input failures.
- [x] Training explicitly has Stage 1, `python train.py --stage 3d`, followed
  by Stage 2, `python train.py --stage dentalpsam`, using the selected 3D
  checkpoint or its prepared features. Current source also supports the public dataset layout and a single weight directory.
- [x] The README contains the full Dataset/Test/Train workflow in fewer than
  150 lines. Advanced details and archived evidence are not prerequisites.
- [x] Reusable training and validation logic lives in packages; validation
  does not import a training CLI. The root interface is `train.py` / `test.py`; only `dentalpsam` is installed.
- [x] Historical checkpoint class names are retained internally; there is no
  architecture, parameter-key, or evaluation-protocol renaming exercise.

These items record a source/document inspection, not a completed runtime gate.

## Verification required on the final candidate

- [x] Run every public command's `--help` without private data.
- [x] Run `bash tests/run_checks.sh` before and after structural changes.
- [x] Run `pytest -q` from a clean checkout.
- [x] Build a wheel containing only `dentalpsam` and distribution metadata.
- [x] Confirm original inputs/outputs and both checkpoint variants remain
  compatible on the designated validation server.
- [x] Scan the Git index for patient artifacts, private paths, and
  credentials, and verify that a clean checkout contains the documented scripts.
- [ ] Have a reader unfamiliar with the code perform the five-minute check below.

Current status: **CPU and server compatibility checks passed; independent reader
check pending**. The [pre-change workflow for 6eea823](https://github.com/HKU-HealthAI/DentalPSAM/actions/runs/36313653347)
passed the original regression command. The
[public-surface workflow for 676e8de](https://github.com/HKU-HealthAI/DentalPSAM/actions/runs/36314319661)
passed editable installation, compilation, 76 pytest checks, all eight original
regression suites, command-help checks, and wheel inspection on a fresh runner.
This includes exact tensor equality through the directory adapter and pinned
numerical-source checks. Model and UV source files were not restructured.
The demonstration program was removed at author request; only unit/regression
test fixtures remain, not a generated-data example or a model-quality claim.
The [29189a4 workflow](https://github.com/HKU-HealthAI/DentalPSAM/actions/runs/36316060775)
also passed, including the packaged SAM license and package guide checks.

After connectivity recovered, revision `29189a4` passed the same 76 tests and
eight regression suites on research23 using the Python 3.9 reference stack.
[Server evidence](verification/release_29189a4.json) records exact native inputs
and zero output differences for gated/concat DentalPSAM and the 3D branch.
The public `test.py --data ... --weights ... --output ...` workflow regenerated
views and 3D features for one real mesh, with predictions exactly matching the
pre-refactor pipeline. For 120 fixed cached meshes, plaque IoU/Dice matched the
native evaluator individually; aggregate metrics and CIs matched the earlier
evaluation. All 240 original PLY hashes remained unchanged. No training,
test-set checkpoint selection, or score-convention change was performed.
Later README wording changes do not alter the tested numerical source.
The separate paper-result gate remains open.

## Commands

Run in the documented environment, from the repository root:

```bash
python -m pip install -e ".[dev]"
python train.py --help
python train.py --stage 3d --help
python train.py --stage dentalpsam --help
python test.py --help
bash tests/run_checks.sh
pytest -q
python dev/check_git_payload.py
```

Preserve logs and the tested Git revision;
do not substitute an earlier successful candidate for the final rerun.
The numerical contract includes architecture/checkpoint keys, mesh ordering,
UV behavior, padding, label and score direction, participant splits, fixed
0.5/0.5 fusion, strict `> 0.5`, and equal-triangle evaluation. Any deviation is
a separate scientific change, not an acceptable side effect of cleanup.

## Five-minute manual check

A researcher who has read the paper but not the code should answer, from the
README and its direct links:

1. Where do I obtain the dataset, and where do its files go?
2. What data and weights do I need to test DentalPSAM, and which command runs it?
3. How do I train the 3D branch?
4. How does its checkpoint feed into DentalPSAM training?
5. Which commands correspond to the paper's method?

Mark this gate passed only when these answers are clear and the final software
checks above pass. A missing project license is intentionally deferred by the
authors in this cleanup; third-party notices remain intact. Repository
acceptance is not a dataset, checkpoint, or license release announcement.
