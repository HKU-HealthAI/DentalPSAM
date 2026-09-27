# Synthetic functionality demo

This synthetic example is provided only for software verification and does not
reproduce the experimental results reported in the paper. It contains no
patient information and does not use pretrained weights, a GPU, or a network.

After installing the package, run from the repository root:

```bash
python examples/synthetic_sample/run.py --output outputs/synthetic_demo
```

The command generates 22 artificial triangles, three prepared views, labels,
mesh patches, and fabricated probability maps. It then checks the directory
contract, constructs the actual model-input tensors through the data loader,
and writes an original-protocol evaluation report. Geometry uses arbitrary
synthetic length units, not clinical measurements.

Read `contracts.json` for shapes and `evaluation/summary.json` for fixture
metrics. Perfect fixture scores and one-sample intervals are expected; they
provide no evidence of segmentation quality. The example does not exercise
anatomy-dependent UV separation or neural inference.
