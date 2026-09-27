# Data pipeline and file contracts

## 1. Original PLY inputs

Each split starts from topology-matched files:

```text
split/
  origin/<mesh_id>.ply   # coloured IOS geometry
  label/<mesh_id>.ply    # same vertices/faces, plaque vertices exactly black
```

Mesh identifiers end in `01` and `02` for the two arches of one participant;
the first four characters are used as the participant identifier in the
current scripts. The TSGCNet implementation expects at most 16,000 faces and
pads a shorter mesh by repeating its last face. The padding is excluded from
loss and evaluation using the original PLY face count.

This repository does not contain the gum-removal or Open3D decimation program
that created the server's 16,000-face PLY inputs. It does not perform uniform
remeshing. Consequently, physical coverage must be measured with original
triangle areas rather than assuming that all faces have equal area.

## 2. Three-view UV projection

`prepare_uv_views.py` uses the historical planar/cylindrical renderer and
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

## 3. TSGCNet mesh features

`export_tsgcnet_features.py` applies a frozen TSGCNet checkpoint. It maps each
UV triangle back to the original PLY face and writes:

```text
tsgc_output/
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

- In `SOTA_mesh`, `value` is TSGCNet class-1 `P(plaque)`.
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
  SOTA_mesh/    # from tsgc_output/SOTA_mesh
  label_mesh/   # from tsgc_output/label_mesh
```

Assemble this as a derived directory or with read-only symlinks. Do not copy
generated files back into the original dataset. Run `tools/preflight_split.py`
before model execution.

## 5. Predictions and reportable evaluation

`predict_dentalpsam.py` writes three continuous 2D probability NPZ/PNG files
and three `3Dpred/<mesh_id>_<view>.npz` files per mesh. Its printed metrics are
branch diagnostics.

`evaluate_dentalpsam.py` is the default MICCAI path. It reprojects the saved
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
