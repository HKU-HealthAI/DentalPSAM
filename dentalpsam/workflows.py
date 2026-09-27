"""Public workflows composing the checked preparation, inference, and evaluator.

No score conventions or numerical kernels are implemented here. The fixed
0.5/0.5 equal-triangle test protocol remains in the evaluation library.
"""

from argparse import Namespace
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

from dentalpsam._layout import PROCESSED_NAMES, SplitLayout


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
    from dentalpsam.branch3d.features import export_features

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


def train_model(args):
    """Prepare fresh inputs with declared weights, then call the trainer."""
    roots = [args.train_dir.resolve(), args.val_dir.resolve()]
    target = (args.out_dir if args.stage == "3d" else args.save_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Choose a new training output: {target}")
    if any(target == root or root in target.parents for root in roots):
        raise ValueError("Training output must be outside the source data")
    branch = getattr(args, "branch_checkpoint", None)
    if args.stage == "dentalpsam" and branch is None:
        raise ValueError("DentalPSAM training requires --branch-checkpoint; unknown cached features are not accepted")
    if args.stage == "dentalpsam" and not args.sam_checkpoint.is_file():
        raise FileNotFoundError(f"SAM initialization not found: {args.sam_checkpoint}")
    if branch is not None and not branch.is_file():
        raise FileNotFoundError(f"3D branch checkpoint not found: {branch}")

    layouts, names = [], []
    for root in roots:
        layout = SplitLayout.read(root)
        names.append(sorted(path.stem for path in layout.labels.glob("*.ply")))
        layouts.append(layout)
    if not all(names):
        raise ValueError(
            "Both training and validation splits must contain labelled meshes"
        )
    if {name[:4] for name in names[0]} & {name[:4] for name in names[1]}:
        raise ValueError("Training and validation participants overlap")

    workspace = None
    if branch is not None or any(layout and layout.public for layout in layouts):
        workspace = fresh_directory(target.with_name(target.name + "_inputs"), roots)
    prepared = []
    preparation_records = {}
    for kind, root, layout, ids in zip(("train", "val"), roots, layouts, names):
        source = (
            layout.legacy_view(
                workspace / f"source_{kind}",
                include_processed=False,
            )
            if layout.public
            else root
        )
        if branch is not None:
            print(f"Preparing {kind} views and frozen 3D features...", flush=True)
            mesh_list = workspace / f"{kind}_mesh_ids.txt"
            mesh_list.write_text("".join(name + "\n" for name in ids))
            source = prepare_split(
                source, mesh_list, branch, workspace / kind, args.device
            )
            from dentalpsam.mesh_io import sha256_file
            export_record = workspace / kind / "features/export_manifest.json"
            record = json.loads(export_record.read_text())
            if record.get("checkpoint_sha256") != sha256_file(branch) or record.get("status") != "completed":
                raise ValueError("Generated training features do not match the declared 3D weights")
            preparation_records[kind] = sha256_file(export_record)
        prepared.append(source if args.stage == "3d" else source / "manual_2D")
    args.train_dir, args.val_dir = prepared
    if args.stage == "3d":
        from dentalpsam.branch3d.training import run_training
    else:
        from dentalpsam.training import run_training
        from dentalpsam.mesh_io import sha256_file
        args.input_provenance = {
            "branch3d_sha256": sha256_file(branch),
            "sam_sha256": sha256_file(args.sam_checkpoint),
            "mesh_fusion": args.mesh_fusion,
            "prepared_export_manifests": preparation_records,
        }
    print(f"Training {'the 3D branch' if args.stage == '3d' else 'DentalPSAM'}: {target}",
          flush=True)
    run_training(args)
    if args.stage == "dentalpsam":
        from dentalpsam.bundle import write_training_bundle_manifest
        write_training_bundle_manifest(target / "manifest.json", target / "best_model.pth",
                                       branch, args.sam_checkpoint)


def test_model(args):
    """Run the complete fixed-list prediction/fusion/evaluation workflow."""
    from dentalpsam.evaluation import read_mesh_ids
    from dentalpsam.inference import predict_split
    from dentalpsam.reporting import evaluate_predictions
    from dentalpsam._preflight import inspect_split
    from dentalpsam.bundle import verify_bundle

    data = args.data.expanduser().resolve()
    bundle = verify_bundle(args.bundle_manifest, args.checkpoint,
                           args.branch_checkpoint, args.sam_checkpoint)
    layout = SplitLayout.read(data)
    mesh_list = args.mesh_list or data / "mesh_ids.txt"
    sam = args.sam_checkpoint or args.checkpoint.parent / "sam_vit_b_01ec64.pth"
    for role, path in (("DentalPSAM checkpoint", args.checkpoint),
                       ("SAM initialization", sam), ("Test list", mesh_list)):
        if not path.is_file():
            raise FileNotFoundError(f"{role} not found: {path}")
    ids = read_mesh_ids(mesh_list)
    if args.expected_count is not None and len(ids) != args.expected_count:
        raise ValueError("Fixed mesh list does not match --expected-count")
    if args.branch_checkpoint is not None and not args.branch_checkpoint.is_file():
        raise FileNotFoundError(f"3D branch checkpoint not found: {args.branch_checkpoint}")
    output = fresh_directory(args.output, [data])
    (output / "bundle_manifest.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(
        f"DentalPSAM: evaluating {len(ids)} meshes; progress log: {output / 'run.log'}",
        flush=True,
    )
    # Keep detailed library output in the log, but show the three main stages
    # on the original console so a long preparation step does not look idle.
    console = sys.stdout
    with (output / "run.log").open("w") as log, redirect_stdout(log):
        def progress(message):
            print(message, file=console, flush=True)
            print(message, flush=True)

        progress("[1/3] Preparing views and 3D features")
        data = layout.legacy_view(output / ".inputs", include_processed=False)
        if args.branch_checkpoint is not None:
            data = prepare_split(
                data,
                mesh_list,
                args.branch_checkpoint,
                output / ".prepared",
                args.device,
            )
            # Expose newly generated inputs under the public directory names.
            (output / "processed").mkdir()
            for old, new in PROCESSED_NAMES.items():
                (output / "processed" / new).symlink_to(
                    data / "manual_2D" / old, target_is_directory=True
                )
        inspect_split(output / "processed", participant_id_prefix_length=4)
        from dentalpsam.mesh_io import sha256_file
        export_record = json.loads((output / ".prepared/features/export_manifest.json").read_text())
        if (export_record.get("status") != "completed"
                or export_record.get("checkpoint_sha256") != bundle["branch3d_sha256"]
                or export_record.get("mesh_list_sha256") != sha256_file(mesh_list)
                or export_record.get("mesh_count") != len(ids)):
            raise ValueError("Generated features do not match the verified weights and fixed mesh list")
        # Catch replacement of a weight file during a long preparation stage.
        verify_bundle(args.bundle_manifest, args.checkpoint, args.branch_checkpoint, sam)
        progress("[2/3] Running DentalPSAM prediction")
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
        progress("[3/3] Evaluating mesh predictions")
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
