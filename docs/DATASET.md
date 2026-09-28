# Dataset

The full dataset is coming soon. Three example scans with matching annotations
are included in [examples/cases](../examples/cases); see the
[example guide](../examples/README.md) for prediction and visualization.

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
    processed/                       # generated views and mesh features
      images/<mesh_id>_{0,1,2}.png
      image_labels/<mesh_id>_{0,1,2}.png
      mesh_features/<mesh_id>.npz
      mesh_labels/<mesh_id>.npz
      metadata/<mesh_id>.npz
```

`train/` and `val/` use the same mesh and annotation layout. Training receives
`--data data`; testing receives `--data data/test`. Testing requires annotations
because the command runs evaluation as well as prediction; this is not an
unlabelled-scan inference interface.

`mesh_ids.txt` is the fixed evaluation list, one six-digit mesh identifier per
line. The first four digits identify a participant, and suffixes `01` / `02`
identify their arches. No cases are automatically added, filtered, or replaced.
Use the evaluation list supplied with the dataset, not a new list assembled
from whichever files happen to be present. Training uses all labelled meshes
in the declared `train/` and `val/` directories rather than `mesh_ids.txt`.
Keep all arches and views from one participant in the same split. Never use
test or external evaluation data for checkpoint or threshold selection.

## Meshes and annotations

`meshes/` contains preprocessed, gingiva-removed coloured PLY meshes;
`labels/` contains matching PLY annotations with exactly the same vertices,
triangles, and ordering. Annotations can contain intermediate grayscale values;
they do not have to be purely black and white. The standalone 3D classifier
and paper evaluator mark a triangle positive only when its per-channel minimum
vertex RGB is `[0, 0, 0]`. On grayscale annotations this means at least one
vertex is exactly black. DentalPSAM's mesh BCE instead retains the continuous
target `1 - min(vertex RGB / 255, per channel)[0]`. Do not replace these soft
training targets with the classifier's binary labels.

Keep the supplied coordinate system and length units. The code does not infer
millimetres from filenames. Annotation geometry and scan geometry must agree.
Gingival removal and mesh simplification happen upstream; they are not run by
this repository. The pipeline does not perform uniform remeshing.

## Prepared inputs

`test.py --weights checkpoints` always generates fresh views and features
using the selected model's 3D checkpoint. Existing `processed/` data are not
read, even when complete. New files are written under the result directory,
never into the original dataset. Stage-2 training likewise requires an explicit
3D checkpoint and regenerates its inputs. There is no public cache-reuse switch.

Use the three weight files supplied together, or from the same training run.
The program strictly loads each model and checks generated labels, face order,
and mesh membership before prediction. No additional model-information file
is required.

| Directory | File contract |
| --- | --- |
| `images` | Three uint8 colour PNGs: upper (`0`, 512 × 768), inner (`1`, 256 × 2048), outer (`2`, 256 × 2048). Loaded as RGB, values 0–255. |
| `image_labels` | Matching PNG masks; grayscale values strictly above 127 are plaque. |
| `mesh_features` | Trusted NPZ with `up`, `in`, `out` object arrays and `face_order_*` arrays. Each face row contains nine triangle XYZ values plus `P(plaque)`. |
| `mesh_labels` | Same patch layout, but channel 9 retains the continuous annotation `1 - min(vertex RGB / 255, per channel)[0]` for mesh BCE. These are targets, never substitutes for input probabilities. |
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

## If a command stops

| Message or symptom | What to check |
| --- | --- |
| Missing test list | Place the supplied `mesh_ids.txt` inside `data/test`. |
| Missing model files | Supply `dentalpsam.pth`, `branch3d.pth`, and `sam.pth`. |
| Incompatible model files | Use the matching files supplied together, or the files from the same training run. |
| Output already exists | Choose a new `--output` directory; previous results are not overwritten. |
| Training and validation participants overlap | Correct the split assignment so every participant belongs to only one split. Do not drop cases based on performance. |

Detailed inference and evaluation messages are saved in `results/run.log`.
