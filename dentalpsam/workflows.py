"""Public workflows composing the checked preparation, inference, and evaluator.

No score conventions or numerical kernels are implemented here. The fixed
0.5/0.5 equal-triangle test protocol remains in the evaluation library.
"""

from argparse import Namespace
from contextlib import redirect_stdout
import json
from pathlib import Path


def fresh_directory(path, sources):
    """Create a new run outside all input trees; never reuse existing output."""
    output = Path(path).expanduser().resolve()
    for source in sources:
        source = Path(source).expanduser().resolve()
        if output == source or source in output.parents:
            raise ValueError("Output must be outside the source data tree")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    return output


def prepare_split(data, mesh_list, checkpoint, output, device):
    """Prepare one fixed split in a fresh directory using frozen 3D weights."""
    from dentalpsam.preparation import prepare_views
    from tsgcnet.features import export_features

    data = Path(data).resolve()
    output = fresh_directory(output, [data])
    prepare_views(
        Namespace(
            origin_dir=data / "origin",
            label_dir=data / "label",
            mesh_list=mesh_list,
            output_dir=output / "views",
        )
    )
    export_features(
        Namespace(
            checkpoint=checkpoint,
            data_dir=data,
            info_dir=output / "views/info",
            mesh_list=mesh_list,
            out_dir=output / "features",
            device=device,
            k=12,
        )
    )
    derived = output / "dataset"
    manual = derived / "manual_2D"
    manual.mkdir(parents=True)
    for kind in ("origin", "label"):
        (derived / kind).symlink_to(data / kind, target_is_directory=True)
    for kind in ("origin", "label", "info"):
        (manual / kind).symlink_to(output / "views" / kind, target_is_directory=True)
    for kind in ("SOTA_mesh", "label_mesh"):
        (manual / kind).symlink_to(output / "features" / kind, target_is_directory=True)
    return derived


def prepare_training_inputs(args):
    """Generate immutable inputs for both declared splits before joint training."""
    roots = [args.train_dir.resolve(), args.val_dir.resolve()]
    names = [
        sorted(path.stem for path in (root / "label").glob("*.ply")) for root in roots
    ]
    if not all(names):
        raise ValueError("Each raw training/validation split needs label/*.ply")
    if {name[:4] for name in names[0]} & {name[:4] for name in names[1]}:
        raise ValueError("Training and validation participants overlap")
    output = fresh_directory(
        args.save_dir.with_name(args.save_dir.name + "_inputs"), roots
    )
    prepared = []
    for kind, source, ids in zip(("train", "val"), roots, names):
        mesh_list = output / f"{kind}_mesh_ids.txt"
        mesh_list.write_text("".join(name + "\n" for name in ids))
        prepared.append(
            prepare_split(
                source, mesh_list, args.branch_checkpoint, output / kind, args.device
            )
            / "manual_2D"
        )
    return tuple(prepared)


def test_model(args):
    """Run the complete fixed-list prediction/fusion/evaluation workflow."""
    from dentalpsam.evaluation import read_mesh_ids
    from dentalpsam.inference import predict_split
    from dentalpsam.reporting import evaluate_predictions
    from tools.preflight_split import inspect_split

    data = args.data.expanduser().resolve()
    mesh_list = args.mesh_list or data / "mesh_ids.txt"
    sam = args.sam_checkpoint or args.checkpoint.parent / "sam_vit_b_01ec64.pth"
    for path in (args.checkpoint, sam, mesh_list):
        if not path.is_file():
            raise FileNotFoundError(path)
    ids = read_mesh_ids(mesh_list)
    if args.expected_count is not None and len(ids) != args.expected_count:
        raise ValueError("Fixed mesh list does not match --expected-count")
    if args.branch_checkpoint is not None and not args.branch_checkpoint.is_file():
        raise FileNotFoundError(args.branch_checkpoint)
    output = fresh_directory(args.output, [data])
    print(
        f"DentalPSAM: evaluating {len(ids)} meshes; progress log: {output / 'run.log'}",
        flush=True,
    )
    with (output / "run.log").open("w") as log, redirect_stdout(log):
        if args.branch_checkpoint is not None:
            data = prepare_split(
                data,
                mesh_list,
                args.branch_checkpoint,
                output / "prepared",
                args.device,
            )
        inspect_split(data / "manual_2D", participant_id_prefix_length=4)
        predict_split(
            Namespace(
                checkpoint=args.checkpoint,
                sam_checkpoint=sam,
                data_dir=data / "manual_2D",
                mesh_list=mesh_list,
                output_dir=output / "predictions",
                expected_count=args.expected_count,
                target_threshold=0.5,
                device=args.device,
                num_workers=args.num_workers,
            )
        )
        evaluate_predictions(
            Namespace(
                data_dir=data,
                prediction_dir=output / "predictions",
                mesh_list=mesh_list,
                output_dir=output / "evaluation",
                expected_count=args.expected_count,
                threshold=0.5,
                fusion_weight_2d=0.5,
                bootstrap_reps=args.bootstrap_reps,
                seed=args.seed,
            )
        )
    metrics = json.loads((output / "evaluation/summary.json").read_text())
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "DentalPSAM Evaluation",
        f"Samples: {len(ids)} meshes / {len({name[:4] for name in ids})} participants",
        "Protocol: equal-triangle; fixed 0.5/0.5 fusion; strict > 0.5",
        "",
    ]
    for label, key in (
        ("Plaque IoU", "plaque_iou"),
        ("Plaque Dice", "plaque_dice"),
        ("Non-plaque IoU", "nonplaque_iou"),
        ("Non-plaque Dice", "nonplaque_dice"),
        ("Accuracy", "accuracy"),
    ):
        value = metrics["fusion"][key]
        lines.append(
            f"{label}: {value['mean']:.4f} [{value['ci_low']:.4f}, {value['ci_high']:.4f}]"
        )
    lines.extend(
        [
            "",
            f"Predictions: {output / 'predictions'}",
            f"Metrics: {output / 'metrics.json'}",
        ]
    )
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    print(summary)
