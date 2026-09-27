"""Read public or historical layouts without renaming any source artifacts.

The numerical loaders retain their checked disk contract. Public workflows
give them a private symlink view in the new run directory. No arrays, mesh
rows, labels, or coordinates are transformed by this adapter.
"""

from dataclasses import dataclass
from pathlib import Path


PROCESSED_NAMES = {
    "origin": "images",
    "label": "image_labels",
    "info": "metadata",
    "SOTA_mesh": "mesh_features",
    "label_mesh": "mesh_labels",
}


@dataclass(frozen=True)
class SplitLayout:
    root: Path
    public: bool

    @classmethod
    def read(cls, root):
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Dataset split does not exist: {root}")
        public = any(
            (root / name).exists() for name in ("meshes", "labels", "processed")
        )
        historical = any(
            (root / name).exists() for name in ("origin", "label", "manual_2D")
        )
        if public and historical:
            raise ValueError(
                "Mixed public and historical directories in one split; use one layout"
            )
        layout = cls(root, public)
        for path in (layout.meshes, layout.labels):
            if not path.is_dir():
                raise NotADirectoryError(
                    f"Required dataset directory is missing: {path}"
                )
        return layout

    @property
    def meshes(self):
        return self.root / ("meshes" if self.public else "origin")

    @property
    def labels(self):
        return self.root / ("labels" if self.public else "label")

    @property
    def processed(self):
        return self.root / ("processed" if self.public else "manual_2D")

    def prepared(self):
        """Reject a partial prepared directory instead of silently regenerating."""
        if not self.processed.exists():
            return False
        for old, new in PROCESSED_NAMES.items():
            path = self.processed / (new if self.public else old)
            if not path.is_dir():
                raise NotADirectoryError(f"Incomplete prepared inputs: {path}")
        return True

    def legacy_view(self, destination, include_processed=True):
        """Return a checked loader path; create links only for the public layout."""
        prepared = self.prepared() if include_processed else False
        if not self.public:
            return self.root
        destination = Path(destination).resolve()
        if destination == self.root or self.root in destination.parents:
            raise ValueError("Compatibility view must be outside the source dataset")
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "origin").symlink_to(self.meshes, target_is_directory=True)
        (destination / "label").symlink_to(self.labels, target_is_directory=True)
        if prepared:
            manual = destination / "manual_2D"
            manual.mkdir()
            for old, new in PROCESSED_NAMES.items():
                (manual / old).symlink_to(
                    self.processed / new, target_is_directory=True
                )
        return destination
