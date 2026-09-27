# Advanced preprocessing

The normal testing and training commands prepare inputs automatically when
given `--3d-checkpoint`. Use the steps below only to inspect individual stages
or prepare a reusable input directory. Preparation uses frozen weights and
does not train either branch.

## Render views and export mesh features

Set absolute paths to an existing preprocessed PLY split, its fixed mesh list,
and the compatible 3D checkpoint. Choose a new directory outside the original
data tree:

```bash
DATA_DIR=/absolute/path/to/test
MESH_LIST=/absolute/path/to/splits/test.txt
FEATURE_CHECKPOINT=/absolute/path/to/checkpoints/3d_branch.pth
PREPARED_DIR="$PWD/outputs/prepared_test"
mkdir -p "$PREPARED_DIR"

python scripts/data/prepare_views.py \
  --origin-dir "$DATA_DIR/origin" --label-dir "$DATA_DIR/label" \
  --mesh-list "$MESH_LIST" --output-dir "$PREPARED_DIR/views"

python scripts/branch3d/export_features.py \
  --checkpoint "$FEATURE_CHECKPOINT" \
  --data-dir "$DATA_DIR" --info-dir "$PREPARED_DIR/views/info" \
  --mesh-list "$MESH_LIST" --out-dir "$PREPARED_DIR/features" \
  --device cuda:0 --k 12
```

## Assemble and check the prepared split

These commands create links only; they do not move or modify the original
PLY files. The filenames follow the [dataset contract](DATASET.md).

```bash
mkdir -p "$PREPARED_DIR/dataset/manual_2D"
ln -s "$DATA_DIR/origin" "$PREPARED_DIR/dataset/origin"
ln -s "$DATA_DIR/label" "$PREPARED_DIR/dataset/label"
ln -s "$PREPARED_DIR/views/origin" "$PREPARED_DIR/dataset/manual_2D/origin"
ln -s "$PREPARED_DIR/views/label" "$PREPARED_DIR/dataset/manual_2D/label"
ln -s "$PREPARED_DIR/views/info" "$PREPARED_DIR/dataset/manual_2D/info"
ln -s "$PREPARED_DIR/features/SOTA_mesh" "$PREPARED_DIR/dataset/manual_2D/SOTA_mesh"
ln -s "$PREPARED_DIR/features/label_mesh" "$PREPARED_DIR/dataset/manual_2D/label_mesh"

python tools/preflight_split.py \
  --data-dir "$PREPARED_DIR/dataset/manual_2D" \
  --participant-id-prefix-length 4 --out "$PREPARED_DIR/input_check.json"
```

## Test the prepared split

```bash
python test.py --data "$PREPARED_DIR/dataset" \
  --mesh-list "$MESH_LIST" \
  --checkpoint checkpoints/dentalpsam.pth \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --output outputs/test_prepared
```

For training, prepare training and validation splits separately using the
same frozen feature checkpoint. Never overwrite inputs used by an existing
run or combine participants across splits. The data specification documents
the renderer's coordinate conventions and preprocessing limitations.
