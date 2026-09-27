#!/usr/bin/env bash
# Run numerical regressions and CLI imports in the selected Python environment.
set -euo pipefail
cd "$(dirname "$0")/.."
validation_python="${1:-python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$validation_python" tests/test_mesh_targets.py
"$validation_python" tests/test_data.py
"$validation_python" tests/test_probability_png.py
"$validation_python" tests/test_sota_inputs.py
"$validation_python" tests/test_prediction_io.py --require_numpy
"$validation_python" tests/test_preflight_split.py
"$validation_python" tests/test_evaluate_mesh.py
"$validation_python" tests/test_original_evaluation.py
for entry in train.py test.py; do
    "$validation_python" "$entry" --help > /dev/null
    printf 'PASS: %s --help\n' "$entry"
done
"$validation_python" train.py --stage 3d --help > /dev/null
"$validation_python" train.py --stage dentalpsam --help > /dev/null
printf 'PASS: both training stages --help\n'
