# Dataset specification

One sample is one dental arch mesh, represented by three rendered views and
aligned triangle features. A participant can contribute two arches. Real scans,
annotations, identities, and split manifests are private and are not included.
The dataset is being organized and is not yet available for public release.
Please reach out to the authors regarding access.

The pipeline expects preprocessed, gingiva-removed meshes, not unprocessed
scanner exports. Gingival removal and mesh simplification are upstream steps;
they are not performed by the preparation commands below.

## Expected layout

Use the existing on-disk contract; no conversion to a new dataset format is
required. Training receives `data/`; testing receives one split, such as
`data/test/`:

```text
data/
  train/
  val/
  test/
    mesh_ids.txt                 # fixed evaluation list, one mesh ID per line
    origin/<mesh_id>.ply          # preprocessed coloured mesh
    label/<mesh_id>.ply           # topology-matched annotation
    manual_2D/                   # prepared model inputs
      origin/<mesh_id>_{0,1,2}.png
      label/<mesh_id>_{0,1,2}.png
      info/<mesh_id>.npz
      SOTA_mesh/<mesh_id>.npz
      label_mesh/<mesh_id>.npz
```

Training and validation splits use the same PLY and prepared-input layout.
`SOTA_mesh` and `label_mesh` are retained filenames in the input contract, not
extra commands that users need to run. Both arches and all views belonging
to one participant must stay in a single train/validation/test split. External
evaluation data must not be used to select checkpoints or training settings.

## Prepare inputs

With prepared `manual_2D/` inputs, use `python test.py` directly as shown in
[Quick start](QUICKSTART.md). For preprocessed PLY pairs without prepared
inputs, add `--3d-checkpoint` to generate views and mesh features automatically.
No retraining is required. The original inputs are not modified; generated
files go into a new output directory outside the source data tree.

Training also prepares inputs automatically when
`python train.py --stage dentalpsam` receives `--3d-checkpoint`. See
[Training](TRAINING.md). For separately running or inspecting preprocessing,
see [Advanced preprocessing](PREPROCESSING.md).

## Array conventions

- PLY files preserve their supplied coordinate system and length units. The
  code does not infer millimeters from a filename. Surface areas have squared
  input-length units. Origin and annotation vertices/faces must correspond.
- Origin PNGs are uint8 RGB when loaded, with values 0–255 (OpenCV storage is
  BGR). Model-input image patches are float32 `[3, 256, 256]`, still in 0–255
  before SAM preprocessing. Binary label patches use `gray > 127`.
- The upper image is `[512, 768]`; inner/outer images are `[256, 2048]`.
  Row-major patch ordering is upper, inner, then outer: 6 + 8 + 8 patches.
- Each mesh NPZ view is a trusted object array of variable-length `[N, 10]`
  floating-point patches. Columns 0–8 are the original triangle vertex XYZ
  coordinates; column 9 is a probability in the input or a binary label in
  the target. Both are left padded with zeros to `[6000, 10]` by the loader.
- UV metadata contains per-vertex floating-point pixel coordinates `[V, 2]`
  and integer vertex-index triples `[F_view, 3]`. Preserve the stored dtype:
  float32 mean rounding can matter at pixel boundaries. Triangles are sorted
  by raster depth within each view, not by original PLY row.
- `face_order_*` is an integer permutation from concatenated patch rows to
  view-face order. Restore this order before saving per-face predictions.
  Padding rows must never enter a score file or a reported metric.
- The 3D feature network repeats the last face to 16,000 rows internally;
  this is different from DentalPSAM's left-zero patch padding. The original
  face count determines which rows are valid at export and evaluation.

The synthetic example generates exactly this layout from artificial geometry;
its generated identifiers and mock scores are not study data.

## 1. Original PLY inputs

Each split starts from topology-matched files:

```text
split/
  origin/<mesh_id>.ply   # coloured IOS geometry
  label/<mesh_id>.ply    # same vertices/faces, plaque vertices exactly black
```

Mesh identifiers end in `01` and `02` for the two arches of one participant;
the first four characters are used as the participant identifier in the
current scripts. The 3D feature implementation expects at most 16,000 faces and
pads a shorter mesh by repeating its last face. The padding is excluded from
loss and evaluation using the original PLY face count.

This repository does not contain the gum-removal or Open3D decimation program
that created the server's 16,000-face PLY inputs. It does not perform uniform
remeshing. Consequently, physical coverage must be measured with original
triangle areas rather than assuming that all faces have equal area.

## 2. Three-view UV projection

`scripts/data/prepare_views.py` uses the historical planar/cylindrical renderer and
writes a fresh staging directory:

```text
uv_output/
  origin/<mesh_id>_{0,1,2}.png
  label/<mesh_id>_{0,1,2}.png
  info/<mesh_id>.npz
  projection_manifest.json
```

View 0 (upper) is 512 x 768 and becomes six 256 x 256 patches. Views 1 and 2
(inner and outer) are 256 x 2048 and each becomes eight patches. Every
`info` NPZ has exactly these keys:

- `uvpx_up`, `uvpx_in`, `uvpx_out`: projected pixel coordinates per vertex;
- `tri_up`, `tri_in`, `tri_out`: the original vertex-index triples assigned to
  each view.

The command checks that the three triangle groups cover the processed PLY face
count. It refuses to overwrite an existing output directory.

The renderer normalizes each observed UV axis using its minimum and maximum.
The historical function signature includes coordinate ranges, but the current
implementation does not use those arguments. Depth sorting controls triangle
overlap in the raster; the stored triangle order is required for reprojection.

## 3. 3D branch input features

The per-face input score is produced by the frozen 3D feature generator,
not by DentalPSAM's mesh encoder or decoder. No additional training is needed
when compatible feature weights or prepared inputs are available.

`scripts/branch3d/export_features.py` applies this frozen checkpoint. It maps each
UV triangle back to the original PLY face and writes:

```text
feature_output/
  SOTA_mesh/<mesh_id>.npz
  label_mesh/<mesh_id>.npz
  SOTA_pred/<mesh_id>_{0,1,2}.png
  scores/<mesh_id>.npy
  export_manifest.json
```

Both mesh NPZ files contain `up`, `in`, `out`, and the corresponding
`face_order_*` arrays. Every non-padding row has ten values:

```text
[x1, y1, z1, x2, y2, z2, x3, y3, z3, value]
```

- In `SOTA_mesh`, `value` is the feature generator's class-1 `P(plaque)`.
- In `label_mesh`, `value` is the binary raw-Ply target: 1 when any incident
  label vertex is exactly black, otherwise 0.
- `face_order_*` restores patch rows to the view's original face order.

The exporter rejects a target-direction disagreement, topology mismatch,
missing requested mesh, malformed probability, duplicate face, or any UV
partition that is not exactly one-to-one with the original PLY faces.

## 4. DentalPSAM input directory

DentalPSAM training and prediction expect one directory containing:

```text
manual_2D/
  origin/       # from uv_output/origin
  label/        # from uv_output/label
  info/         # from uv_output/info, required for final evaluation
  SOTA_mesh/    # from feature_output/SOTA_mesh
  label_mesh/   # from feature_output/label_mesh
```

Assemble this as a derived directory or with read-only symlinks. Do not copy
generated files back into the original dataset. Run `tools/preflight_split.py`
before model execution.

## 5. Predictions and reportable evaluation

`scripts/dentalpsam/predict.py` writes three continuous 2D probability NPZ/PNG files
and three `3Dpred/<mesh_id>_<view>.npz` files per mesh. Its printed metrics are
branch diagnostics.

`scripts/dentalpsam/evaluate.py` is the default MICCAI path. It reprojects the saved
8-bit PNG probabilities using truncated UV centres, applies fixed 0.5/0.5
fusion with the 3D scores, thresholds strictly above 0.5, and computes
equal-triangle metrics per mesh. Its point estimate is the mean over meshes;
its confidence intervals resample participants with all their meshes.

`tools/evaluate_mesh.py` is the separate physical-area analysis. It maps every saved score to
the original PLY face, combines arches `01` and `02`, computes each patient's
metrics, and then reports the patient macro mean with patient-resampled 95%
bootstrap confidence intervals. Area metrics weight each face by

```text
0.5 * ||(v2 - v1) x (v3 - v1)||.
```

Paired model comparisons use patient-level differences, paired-bootstrap
intervals, two-sided sign-flip tests, and Holm adjustment.
