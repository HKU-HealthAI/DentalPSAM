"""Optional mesh-feature augmentation; disabled by the reference 3D trainer."""

import numpy as np
import random


class DataAugmentation:
    """Transform geometry and normals together, preserving scan colours."""

    def __init__(
        self,
        enable_flip=True,
        flip_prob=0.5,
        enable_rotation=False,
        rotation_range=(-10, 10),
        seed=None,
    ):
        """Configure reflection probability and Z-axis rotation range in degrees."""
        self.enable_flip = enable_flip
        self.flip_prob = flip_prob
        self.enable_rotation = enable_rotation
        self.rotation_range = rotation_range

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def vertical_flip(self, points_face):
        """Reflect Y coordinates and Y normals in an [N, 33] feature array."""
        points_face_flipped = points_face.copy()

        # Three vertices and the face centre.
        points_face_flipped[:, 1] *= -1
        points_face_flipped[:, 4] *= -1
        points_face_flipped[:, 7] *= -1
        points_face_flipped[:, 10] *= -1

        # Normal vectors follow the same reflection. RGB is unchanged.
        points_face_flipped[:, 13] *= -1
        points_face_flipped[:, 16] *= -1
        points_face_flipped[:, 19] *= -1

        return points_face_flipped

    def random_rotation_z(self, points_face, angle_range=None):
        """Apply one sampled Z-axis rotation to vertices, centre and normals."""
        if angle_range is None:
            angle_range = self.rotation_range

        angle = np.random.uniform(angle_range[0], angle_range[1])
        theta = np.radians(angle)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot_matrix = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )

        points_face_rotated = points_face.copy()

        for i in range(3):
            start_idx = i * 3
            end_idx = start_idx + 3
            coords = points_face_rotated[:, start_idx:end_idx]
            points_face_rotated[:, start_idx:end_idx] = coords @ rot_matrix.T

        centre = points_face_rotated[:, 9:12]
        points_face_rotated[:, 9:12] = centre @ rot_matrix.T

        for i in range(3):
            start_idx = 12 + i * 3
            end_idx = start_idx + 3
            normal = points_face_rotated[:, start_idx:end_idx]
            points_face_rotated[:, start_idx:end_idx] = normal @ rot_matrix.T

        return points_face_rotated

    def augment(self, points_face, is_training=True):
        """Apply enabled transforms only during training; preserve RNG order."""
        if not is_training:
            return points_face

        if self.enable_flip and np.random.random() < self.flip_prob:
            points_face = self.vertical_flip(points_face)

        # Retain the original 30% rotation probability.
        if self.enable_rotation and np.random.random() < 0.3:
            points_face = self.random_rotation_z(points_face)

        return points_face

    def get_augmentation_info(self):
        """Return the configured augmentation options."""
        return {
            "enable_flip": self.enable_flip,
            "flip_prob": self.flip_prob,
            "enable_rotation": self.enable_rotation,
            "rotation_range": self.rotation_range,
        }
