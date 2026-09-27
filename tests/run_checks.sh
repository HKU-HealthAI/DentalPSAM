#!/usr/bin/env bash
# Run synthetic regressions and CLI imports in the selected Python environment.
set -euo pipefail
cd "$(dirname "$0")/.."
validation_python="${1:-python}"
export PYTHONDONTWRITEBYTECODE=1
"$validation_python" tests/test_mesh_targets.py
"$validation_python" tests/test_prediction_io.py --require_numpy
"$validation_python" tests/test_preflight_split.py
"$validation_python" tests/test_evaluate_mesh.py
"$validation_python" tests/test_original_evaluation.py
for entry in prepare_uv_views.py train_tsgcnet.py validate_tsgcnet.py \
    export_tsgcnet_features.py train_dentalpsam.py validate_dentalpsam.py \
    predict_dentalpsam.py evaluate_dentalpsam.py tools/evaluate_mesh.py \
    tools/verify_server_compatibility.py; do
    "$validation_python" "$entry" --help > /dev/null
    printf 'PASS: %s --help\n' "$entry"
done
