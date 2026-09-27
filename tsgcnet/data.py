"""Historical 16,000-face TSGCNet inputs with checkpoint-sensitive normalization.

Feature columns are XYZ vertices (0:9), face centre (9:12), vertex normals
(12:21), minimum face RGB (21:24), and vertex RGB (24:33). Label geometry
supplies connectivity/normals; colors come from the matching origin PLY, never
from the plaque annotation. Short meshes repeat the final face to 16,000 rows;
callers must exclude padding from loss, exported scores, and evaluation.
"""

from plyfile import PlyData
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import copy
import open3d as o3d
from .augmentation import DataAugmentation


def load_ply_features(path=""):
    """Return face indices, 33 features, class labels, one-hot labels, and vertices.

    Black annotation vertices define plaque/class 1 through the historical
    per-channel minimum rule. Use topology-matched origin/label PLY files.
    This routine does not downsample, equalize triangle areas, or remesh.
    """
    labels = [0, 0, 0]
    row_data = PlyData.read(path)
    points = np.array(pd.DataFrame(row_data.elements[0].data))
    faces = np.array(pd.DataFrame(row_data.elements[1].data))

    n_face = faces.shape[0]

    # Repeat the final face without changing original face order.
    while faces.shape[0] < 16000:

        N = copy.deepcopy(faces[-1])
        N = N.reshape(1, -1)
        faces = np.concatenate((faces, N), axis=0)
        n_face += 1

    xyz = points[:, :3]

    components = path.split("/")
    new_path = "/".join(components[:-2])
    plyname = components[-1]

    origin_path = os.path.join(new_path, "origin")

    origin_path = os.path.join(origin_path, plyname)
    origin_data = PlyData.read(origin_path)

    origin_points = np.array(pd.DataFrame(origin_data.elements[0].data))

    if origin_points.shape[1] == 6:
        color = origin_points[:, 3:6] / 255
    else:
        color = origin_points[:, 6:9] / 255
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    label_face = np.zeros([n_face, 1]).astype("int32")
    label_face_onehot = np.zeros([n_face, 2]).astype(("int32"))

    index_face = (
        np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3).astype("int32")
    )

    RGB_face = np.zeros((n_face, 3))
    for i in range(0, n_face):

        indices = faces[i][0]

        if points.shape[1] == 6:
            colors = points[indices, 3:6]
        else:
            colors = points[indices, 6:9]

        min_color = np.min(colors, axis=0)

        RGB_face[i] = min_color

    max_vertex_index = len(xyz) - 1
    max_normal_index = len(normals) - 1
    max_color_index = len(color) - 1

    max_face_index = np.max(index_face)
    if (
        max_face_index > max_vertex_index
        or max_face_index > max_normal_index
        or max_face_index > max_color_index
    ):
        print(f"警告: 文件 {path} 中索引超出范围")
        print(f"  面片最大索引: {max_face_index}")
        print(f"  顶点数组大小: {max_vertex_index + 1}")
        print(f"  法向量数组大小: {max_normal_index + 1}")
        print(f"  颜色数组大小: {max_color_index + 1}")

        valid_mask = (
            (index_face[:, 0] <= max_vertex_index)
            & (index_face[:, 1] <= max_vertex_index)
            & (index_face[:, 2] <= max_vertex_index)
        )
        valid_mask &= (
            (index_face[:, 0] <= max_normal_index)
            & (index_face[:, 1] <= max_normal_index)
            & (index_face[:, 2] <= max_normal_index)
        )
        valid_mask &= (
            (index_face[:, 0] <= max_color_index)
            & (index_face[:, 1] <= max_color_index)
            & (index_face[:, 2] <= max_color_index)
        )

        n_valid_faces = np.sum(valid_mask)
        if n_valid_faces == 0:
            print(f"错误: 文件 {path} 没有有效的面片")
            return None, None, None, None, None

        print(f"保留 {n_valid_faces}/{n_face} 个有效面片")
        index_face = index_face[valid_mask]
        faces = faces[valid_mask]
        n_face = n_valid_faces

        label_face = np.zeros([n_face, 1]).astype("int32")
        label_face_onehot = np.zeros([n_face, 2]).astype("int32")
        RGB_face = np.zeros((n_face, 3))

    try:

        xyz_face = np.concatenate(
            (
                xyz[index_face[:, 0], :],
                xyz[index_face[:, 1], :],
                xyz[index_face[:, 2], :],
            ),
            axis=1,
        )

        normal_vertex = np.concatenate(
            (
                normals[index_face[:, 0], :],
                normals[index_face[:, 1], :],
                normals[index_face[:, 2], :],
            ),
            axis=1,
        )
        color_vertex = np.concatenate(
            (
                color[index_face[:, 0], :],
                color[index_face[:, 1], :],
                color[index_face[:, 2], :],
            ),
            axis=1,
        )
    except Exception as e:
        print(f"错误: 文件 {path} 在处理面片数据时出错: {e}")
        return None, None, None, None, None

    x1, y1, z1 = xyz_face[:, 0], xyz_face[:, 1], xyz_face[:, 2]
    x2, y2, z2 = xyz_face[:, 3], xyz_face[:, 4], xyz_face[:, 5]
    x3, y3, z3 = xyz_face[:, 6], xyz_face[:, 7], xyz_face[:, 8]
    x_centre = (x1 + x2 + x3) / 3
    y_centre = (y1 + y2 + y3) / 3
    z_centre = (z1 + z2 + z3) / 3
    centre_face = np.concatenate(
        (
            x_centre.reshape(n_face, 1),
            y_centre.reshape(n_face, 1),
            z_centre.reshape(n_face, 1),
        ),
        axis=1,
    )

    RGB = np.zeros([n_face, 3])

    for i in range(0, n_face):

        id0 = faces[i][0][0]
        id1 = faces[i][0][1]
        id2 = faces[i][0][2]
        RGB[i] = [
            min(color[id0][0], color[id1][0], color[id2][0]),
            min(color[id0][1], color[id1][1], color[id2][1]),
            min(color[id0][2], color[id1][2], color[id2][2]),
        ]

    points_face = np.concatenate(
        (xyz_face, centre_face, normal_vertex, RGB, color_vertex), axis=1
    ).astype("float32")

    label_face[(RGB_face == labels).all(axis=1)] = 1
    label_face_onehot[(RGB_face == labels).all(axis=1), 1] = 1
    label_face_onehot[(RGB_face != labels).all(axis=1), 0] = 1

    return index_face, points_face, label_face, label_face_onehot, origin_points


def save_face_predictions(index_face, point_face, label_face, path=" "):
    """
    Input:
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
        label_face: label of face [N, 1]
        path: path to save new generated ply file
    Return:
    """

    labels_change_color = [[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    name = path[-10:]
    pred_face_path = os.path.join(path[:-10], "pred_face")
    index_face_path = os.path.join(path[:-10], "index_face")
    if not os.path.exists(pred_face_path):

        os.makedirs(pred_face_path, exist_ok=True)
        os.makedirs(index_face_path, exist_ok=True)
    np.save(os.path.join(pred_face_path, name[:-4] + ".npy"), label_face.numpy())
    np.save(os.path.join(index_face_path, name[:-4] + ".npy"), index_face)


class PlyDataset(Dataset):
    """Apply legacy normalization without changing a frozen checkpoint's inputs."""

    def __init__(self, path="data/train", enable_augmentation=False, flip_prob=0.5):
        self.root_path = path
        self.file_list = [f for f in os.listdir(path) if f.endswith(".ply")]

        self.enable_augmentation = enable_augmentation
        if enable_augmentation:
            self.augmenter = DataAugmentation(
                enable_flip=True,
                flip_prob=flip_prob,
                enable_rotation=False,
                rotation_range=(-10, 10),
            )
            print(f"数据增强已启用: 翻转概率={flip_prob}")
        else:
            self.augmenter = None
            print("数据增强已禁用")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_face, label_face, label_face_onehot, points = (
            load_ply_features(path=read_path)
        )
        raw_points_face = points_face.copy()

        if self.enable_augmentation and self.augmenter is not None:

            points_face = self.augmenter.augment(points_face, is_training=True)

        # Compatibility: the center includes repeated padding faces.
        centre = points_face[:, 9:12].mean(axis=0)
        points_face[:, 0:3] -= centre
        points_face[:, 3:6] -= centre
        points_face[:, 6:9] -= centre
        points_face[:, 9:12] = (
            points_face[:, 0:3] + points_face[:, 3:6] + points_face[:, 6:9]
        ) / 3
        points[:, :3] -= centre
        # This scalar maximum includes original color channels, not XYZ alone.
        coordinate_scale = points.max()
        points_face[:, :12] = points_face[:, :12] / coordinate_scale

        # Use original-vertex statistics, not statistics of scaled face features.
        maxs = points[:, :3].max(axis=0)
        mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = points[:, :3].std(axis=0)
        nmeans = points[:, 3:].mean(axis=0)
        nstds = points[:, 3:].std(axis=0)
        nmeans_f = points_face[:, 21:24].mean(axis=0)
        nstds_f = points_face[:, 21:24].std(axis=0)
        for i in range(3):

            points_face[:, i] = (points_face[:, i] - means[i]) / stds[i]
            points_face[:, i + 3] = (points_face[:, i + 3] - means[i]) / stds[i]
            points_face[:, i + 6] = (points_face[:, i + 6] - means[i]) / stds[i]
            points_face[:, i + 9] = (points_face[:, i + 9] - mins[i]) / (
                maxs[i] - mins[i]
            )

        return (
            index_face,
            points_face,
            label_face,
            label_face_onehot,
            self.file_list[item],
            raw_points_face,
        )


# Historical aliases for external callers; new code uses the names above.
get_data = load_ply_features
generate_plyfile = save_face_predictions
plydataset = PlyDataset
