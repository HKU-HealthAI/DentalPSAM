"""Bind the three weight files to an author-declared model bundle.

Hashes detect substitution, not authorship or training history. Obtain historical
manifests from the authors; do not manufacture a manifest to bypass a mismatch.
New training checkpoints also carry their input-weight hashes internally.
"""

import json
from pathlib import Path
import re

from dentalpsam.mesh_io import sha256_file


WEIGHT_KEYS = ("dentalpsam_sha256", "branch3d_sha256", "sam_sha256")
WEIGHT_LABELS = dict(zip(WEIGHT_KEYS, ("DentalPSAM", "3D branch", "SAM")))
WEIGHT_LABELS["mesh_fusion"] = "fusion architecture"


def checkpoint_binding(checkpoint):
    """Read architecture and optional training binding from trusted weights."""
    import torch
    from dentalpsam.checkpoints import extract_model_state

    value = torch.load(checkpoint, map_location="cpu")
    state = extract_model_state(value)
    fusion = "concat" if any(key.startswith("mesh_decoder.mlp_fusion.") for key in state) else "gated"
    return fusion, value.get("input_provenance")


def verify_bundle(manifest_path, dentalpsam, branch3d, sam):
    """Fail before preprocessing if files, declared hashes or architecture differ."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Model information missing: {manifest_path}. Keep manifest.json "
            "with the matching weight files supplied by the authors or training run."
        )
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("model") != "DentalPSAM":
        raise ValueError("Unsupported model information; use the manifest.json supplied with your weights")
    if manifest.get("mesh_fusion") not in ("gated", "concat"):
        raise ValueError("Model information must specify gated or concat fusion")
    provenance = manifest.get("provenance", {})
    if not isinstance(provenance, dict) or provenance.get("kind") not in ("training_checkpoint", "author_verified"):
        raise ValueError("Model information is incomplete; obtain the complete files from the authors or training run")
    if not isinstance(provenance.get("reference"), str) or not provenance["reference"].strip():
        raise ValueError("Model information is missing its training or author reference")
    for key, path in zip(WEIGHT_KEYS, (dentalpsam, branch3d, sam)):
        expected = manifest.get(key)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Invalid model information for {WEIGHT_LABELS[key]}; restore the supplied manifest.json")
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"{WEIGHT_LABELS[key]} weights missing: {path}")
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"Model files do not match: {WEIGHT_LABELS[key]}; use the matching files supplied together")
    fusion, binding = checkpoint_binding(dentalpsam)
    if fusion != manifest["mesh_fusion"]:
        raise ValueError("Model information differs from checkpoint architecture")
    if provenance["kind"] == "training_checkpoint" and not isinstance(binding, dict):
        raise ValueError("Checkpoint is missing the expected training information")
    if binding is not None:
        for key in ("branch3d_sha256", "sam_sha256", "mesh_fusion"):
            if not isinstance(binding, dict) or binding.get(key) != manifest[key]:
                raise ValueError(f"Training inputs do not match the model files: {WEIGHT_LABELS[key]}")
    return manifest


def write_training_bundle_manifest(output, dentalpsam, branch3d, sam):
    """Export a manifest only for weights carrying a matching training binding."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fusion, binding = checkpoint_binding(dentalpsam)
    if not isinstance(binding, dict):
        raise ValueError("Checkpoint has no saved input information; obtain the matching manifest.json from the authors")
    manifest = {
        "schema_version": 1, "model": "DentalPSAM", "mesh_fusion": fusion,
        **{key: sha256_file(Path(path)) for key, path in
           zip(WEIGHT_KEYS, (dentalpsam, branch3d, sam))},
        "provenance": {"kind": "training_checkpoint", "reference": "input_provenance in dentalpsam.pth"},
    }
    for key in ("branch3d_sha256", "sam_sha256", "mesh_fusion"):
        if binding.get(key) != manifest[key]:
            raise ValueError(f"Training inputs do not match the model files: {WEIGHT_LABELS[key]}")
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
