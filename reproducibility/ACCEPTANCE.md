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

- [ ] Run every public command's `--help` without private data.
- [ ] Run `bash tests/run_checks.sh` before and after structural changes.
- [ ] Run `pytest -q` from a clean checkout.
- [ ] Confirm original inputs/outputs and both checkpoint variants remain
  compatible on the designated validation server.
- [ ] Scan the final Git index for patient artifacts, private paths, and
  credentials, and verify that a clean checkout contains the documented scripts.
- [ ] Have a reader unfamiliar with the code perform the five-minute check below.

Current status: **pending final verification**, not accepted. Earlier server
checks covered eight regression suites; an intermediate structural candidate
also passed 53 pytest checks. These do not certify the final source revision.
The [CPU workflow for 1d75106](https://github.com/HKU-HealthAI/DentalPSAM/actions/runs/36313424733)
passed editable installation, compilation, and pytest on a fresh GitHub runner.
This does not replace private-checkpoint server checks. The demonstration
program from that earlier revision was subsequently removed at author request.
The SSH jump route was unavailable during the latest final-candidate attempt.
No training or paper-metric optimization was launched to work around it.

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
