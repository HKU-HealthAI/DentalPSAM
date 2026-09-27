# Dataset

The dataset is being organized and is not publicly released yet. Please reach
out to the authors regarding data access. No patient scans, annotations,
identifiers, or cohort manifests are included in this repository.

## Input layout

One sample is a dental arch mesh. A participant may contribute two arches.
Use this structure for each split:

```text
data/
  train/
  val/
  test/
    mesh_ids.txt
    meshes/<mesh_id>.ply
    labels/<mesh_id>.ply
    processed/                       # optional prepared inputs
      images/<mesh_id>_{0,1,2}.png
      image_labels/<mesh_id>_{0,1,2}.png
      mesh_features/<mesh_id>.npz
      mesh_labels/<mesh_id>.npz
      metadata/<mesh_id>.npz
```

`train/` and `val/` use the same mesh and annotation layout. Training receives
`--data data`; testing receives `--data data/test`.

`mesh_ids.txt` is the fixed evaluation list, one six-digit mesh identifier per
line. The first four digits identify a participant, and suffixes `01` / `02`
identify their arches. No cases are automatically added, filtered, or replaced.
Keep all arches and views from one participant in the same split. Never use
test or external evaluation data for checkpoint or threshold selection.

## Meshes and annotations

`meshes/` contains preprocessed, gingiva-removed coloured PLY meshes;
`labels/` contains matching PLY annotations with exactly the same vertices,
triangles, and ordering. Plaque vertices are exactly black. The original
triangle target uses the per-channel minimum of its vertex colours; binary
black/white annotations make any black vertex a positive triangle.

Keep the supplied coordinate system and length units. The code does not infer
millimetres from filenames. Annotation geometry and scan geometry must agree.
Gingival removal and mesh simplification happen upstream; they are not run by
this repository. The pipeline does not perform uniform remeshing.

## Prepared inputs

When `processed/` is absent, `test.py --weights checkpoints` generates the
necessary views and features using the fixed 3D checkpoint. New files are
written under the result directory, never into the original dataset.
When prepared inputs are supplied, all five subdirectories must be complete;
a partial directory is an error, not a request to silently regenerate it.

Prepared features must come from the checkpoint associated with the trained
DentalPSAM model. A different 3D checkpoint is not an interchangeable input.

| Directory | File contract |
| --- | --- |
| `images` | Three uint8 colour PNGs: upper (`0`, 512 × 768), inner (`1`, 256 × 2048), outer (`2`, 256 × 2048). Loaded as RGB, values 0–255. |
| `image_labels` | Matching PNG masks; grayscale values strictly above 127 are plaque. |
| `mesh_features` | Trusted NPZ with `up`, `in`, `out` object arrays and `face_order_*` arrays. Each face row contains nine triangle XYZ values plus `P(plaque)`. |
| `mesh_labels` | Same patch layout, but channel 9 contains the binary annotation. These are targets, never substitutes for input probabilities. |
| `metadata` | `uvpx_up/in/out`: vertex pixel coordinates `[V, 2]`; `tri_up/in/out`: integer vertex-index triples `[F_view, 3]`. Preserve stored dtypes and order. |

Each view becomes row-major 256 × 256 patches: 6 upper, 8 inner, 8 outer.
The loader left-pads each variable-length `[N, 10]` mesh patch with zeros to
`[6000, 10]`. Padding is excluded from reported metrics. The 3D feature model
separately repeats the final face to 16,000 rows; only original faces are
exported. `face_order_*` restores patch rows to view-face order before saving
predictions. Never sort or resample mesh rows independently.

Only load trusted checkpoint and NPZ files; the retained formats use pickle.

## Compatibility and outputs

Historical directory names are recognized internally. Existing datasets do
not need to be renamed. A private symlink view in the new run directory adapts
public names to the checked numerical loaders; it does not transform arrays.
Do not mix public and historical names within one split.

The output directory must be new and outside the input tree. Testing writes
`predictions/`, `metrics.json`, `summary.txt`, and an evaluation manifest.
The single default evaluator is the original equal-triangle paper protocol:
fixed 0.5/0.5 fusion, strict `> 0.5`, and mesh-macro estimates with
participant-clustered confidence intervals. It is not area-weighted evaluation.
This software contract does not itself establish reproduction of paper values.
