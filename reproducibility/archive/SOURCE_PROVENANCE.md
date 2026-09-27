# Source provenance

Archived source-inspection record. Filenames below describe the inspected
sources and pre-refactor entry points, not the current public command interface.
See the repository README for testing and two-stage training.

The public tree was prepared from read-only snapshots on `research23` on
2026-09-27. Private absolute paths were removed from executable entry points.
The following SHA-256 values bind the upstream files that were inspected.

## MICCAI figures and README results

The README uses the author-supplied `MICCAI2026_DentalPSAM (1).zip` archive,
not the CVPR archive or clinical tables. Its `Paper-4344.pdf` (pages 2 and 7)
and `LatexSource-4344/sec/experiment.tex` were checked together. README metric
values are a transcription of active Table 1, not the commented-out earlier
table. They are paper reports, not evidence of a successful current replay.

Only the two paper illustrations are distributed, not the paper archive,
publication agreement, patient scans, annotations, or cohort identifiers.
The PNGs are direct single-page renders of the original figure PDFs; no
predictions, labels, or figure content were edited.

| Artifact | SHA-256 |
| --- | --- |
| Supplied MICCAI archive | `56895e49bd2a09c9626fa5810ef19060f7f9af184b527c33b19ff3e87095d6f5` |
| `Paper-4344.pdf` | `786581e4629d0f1e530444aca62da479c2821bbf7b4f3e21c1d6aa420c07596e` |
| `images/pipe_dentalpsam.pdf` | `2015f334fd7f329876f7c2bd02a3d991156fe31b48ee487a6272989fab14ace8` |
| `images/experiments_visuliazation.pdf` | `d2331b6e93466464e7e11851f57221349e82829d6f05a17687c241fa5ab5ed79` |
| `assets/method.png` | `24b1172b7c87765888f49e1e06d5b30714d77391b33fad4fdf420f73e0ea19cc` |
| `assets/comparison.png` | `3e83d8c58994babf8099b4e230ba28b3480661f822c51145a795d906ea751083` |

Rendering used `pdftoppm -singlefile -png -scale-to 1800` for the architecture
and `-scale-to 2000` for the comparison. Figures retain their paper labels;
the comparison baseline names are not additional public workflow commands.

## DentalPSAM server source

| Upstream file | SHA-256 |
| --- | --- |
| `raw_code/SAMoral/model.py` | `dc2fe0902b9205188b03791eecffcce4caf9c5106e500f8f1893f90070213b9c` |
| Historical `SAMoral/model.py.backup_20260215` (preserved June source archive) | `e928750dd9279eab2904a045133089409200b8b2b3cc5c06a1448ced0b47ad3b` |
| `raw_code/SAMoral/train.py` | `010ec8de0e4df99073716ac935e8064386dc56fa02d33c6387f370fd4ce17153` |
| `raw_code/SAMoral/evaluation.py` | `edb37ad04dd3c437e5fb590abdaa79c14679c1536ca8351b2d56db0d11270536` |
| `raw_code/SAMoral/pred2D_jhn.py` | `079204c4f952e76a079a1d04a979aa74a2cbf85373cb8da3cc28c8a650305c45` |
| `raw_code/SAMoral/dataloader.py` | `ef80fee42d513254cdf4a1b859ce81caf671553d2358f3acd3c2cef62addd257` |
| `raw_code/SAMoral/iou.py` | `2f908f4ffdddb797ad844ea52d7d7d2f19f17ba63fc1ee4d7aaaecf714524212` |
| `raw_code/evaluation3d_confidence_fusion.py` | `449ff61d96b90f07dda8d19fa21e5f67766c0a39fe21dc4d3baf9d659aeb5bc8` |

## TSGCNet and UV source

| Upstream file | SHA-256 |
| --- | --- |
| `TSGCNet-main_cnc_new_true/TSGCNet.py` | `6fe2b71f39d25b4458d962af3f7f7b88e9aa0e46c9fc528344767e83b7e24ce9` |
| `TSGCNet-main_cnc_new_true/dataloader.py` | `c402c50db337da2a13a213880f931e66e1ff340c7a48146cb67f4607163f7c9c` |
| `TSGCNet-main_cnc_new_true/utils.py` | `e71b438466a443545ee4d66a5c3940e4319b845a7a70dd70b9faf1b3e7b31597` |
| `TSGCNet-main_cnc_new_true/train.py` | `d134ea389c947a53d4aec2a54b316bc24885825cc95909fd798b545b131f61e6` |
| `TSGCNet-main_cnc_new_true/evaluation.py` | `2d7adf681daa9f49a385480f02d687672a4498b38057bb1a0aadc8da4aa04fda` |
| `TSGCNet-main_cnc_new_true/test_only.py` | `051e787207ac097ad34eb9db6e7897bd420049c4e15ce8eb467abdb3f1b2d7ad` |
| `TSGCNet-main_cnc_new_true/pred_to_2d_new.py` | `ad1da89396f32d6d97c2c03535617863d98a1c0057939b7a6d564431c9e77039` |
| `TSGCNet-main_cnc_new_true/data_augmentation.py` | `a8655e2233e62cb5320d48a25d735155c7ffa257cd11e5b928a4df45362d2da6` |
| `uvraster.py` | `16bb6c42b5accd07984f88b1c1dd8b81d72d4c52e9d2fb8baff2d88e96b89028` |

## Deliberate release changes

There are two distinct DentalPSAM architectures, not one interchangeable
`model.py`. The historical backup uses multiplicative gating (397 checkpoint
keys); the newer server source uses concatenation plus `mlp_fusion` (403 keys).
The package supports both. Frozen-checkpoint loading selects the structure by
the `mesh_decoder.mlp_fusion.*` keys, then loads **strictly**. A missing fusion
head is never randomly initialized to make a checkpoint load. Training declares
the variant with `--mesh-fusion gated|concat`; its historical default is gated.
Neither variant alone proves which checkpoint produced the MICCAI table.

- The package names are standardized as `dentalpsam` and `tsgcnet`.
- All data, checkpoint, GPU, and output paths are explicit CLI arguments.
- Training, validation, prediction, preprocessing, and evaluation are separate
  entry points.
- Training and prediction use the same binary 2D target rule (`gray > 127`).
- SAM is moved to the inference device before wrapping it: its bound
  preprocessing method retains the SAM-owned pixel-mean/std buffers.
- Zero-padded mesh rows are excluded from DentalPSAM 3D loss and diagnostics.
- TSGCNet selection uses only the declared validation split; test and external
  directories are not accepted by the training command.
- The unsafe historical `pred_to_2d_new.py` is replaced by
  `export_tsgcnet_features.py`, which writes both `SOTA_mesh` and `label_mesh`,
  checks the exact UV-face partition, and records hashes.
- `evaluate_dentalpsam.py` preserves the inspected fixed-fusion evaluator's
  original equal-triangle protocol, while adding strict input checks and
  participant-clustered confidence intervals.
- `tools/evaluate_mesh.py` is a separate area/face/vertex audit with different
  aggregation and projection rules; it is not the paper-reproduction default.

These changes intentionally make the experiment protocol stricter. Therefore,
an old checkpoint may be evaluated compatibly, but a newly trained checkpoint
is not claimed to be bit-identical to an undocumented historical run.
