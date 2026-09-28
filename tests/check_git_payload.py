#!/usr/bin/env python3
"""Reject private artifacts, oversized files, and common secrets in Git's index.

This check prints only offending paths, never matched secret content. It is a
safeguard alongside human review, not a guarantee that all secrets are found.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    forbidden_suffixes = {".ply", ".npy", ".npz", ".pth", ".pt", ".h5", ".hdf5", ".pkl",
                          ".zip", ".tar", ".gz", ".bundle", ".pem", ".key",
                          ".json", ".jsonl", ".sha256"}
    forbidden_dirs = {"data", "Raw", "derived", "outputs", "checkpoints", "results",
                      "local_evidence", ".venv", "venv"}
    secret = re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
    failures = []
    checked = 0
    for name in filter(None, paths):
        path = Path(name)
        example_asset = bool(re.fullmatch(
            r"examples/(?:cases/(?:meshes|labels)/000[123]01\.ply|"
            r"results/processed/metadata/000[123]01\.npz|"
            r"results/predictions/000[123]01_[012]\.png|"
            r"results/predictions/3Dpred/000[123]01_[012]\.npz|"
            r"results/meshes/000[123]01\.ply|"
            r"previews/000[123]01\.png)", name
        ))
        reason = None
        if not example_asset and (path.suffix.lower() in forbidden_suffixes or (
            set(path.parts) & forbidden_dirs
        )):
            reason = "private/generated artifact"
        if path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example":
            reason = "environment credentials"
        blob = subprocess.check_output(["git", "show", ":" + name], cwd=root)
        size_limit = (4 if example_asset else 1) * 1024 * 1024
        if len(blob) > size_limit:
            reason = "file exceeds repository size budget"
        if secret.search(blob):
            reason = "credential pattern"
        if reason:
            failures.append(f"{name}: {reason}")
        checked += 1
    if failures:
        raise SystemExit("Git payload check failed:\n" + "\n".join(failures))
    print(f"PASS: {checked} indexed files; no forbidden artifacts or detected credential patterns")


if __name__ == "__main__":
    main()
