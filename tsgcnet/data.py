from plyfile import PlyData
import numpy as np
from torch.utils.data import DataLoader,Dataset,random_split
import os
import pandas as pd
import copy
import open3d as o3d
from .augmentation import DataAugmentation



def load_ply_features(path=""):
    labels = ([0, 0, 0])
    row_data = PlyData.read(path)  # read ply file
    points = np.array(pd.DataFrame(row_data.elements[0].data)) # [?,?,?,x,y,z,r,g,b] for data_teethseg3w
    faces = np.array(pd.DataFrame(row_data.elements[1].data))
    # print('faces', faces.shape) #(16000, 1)
    n_face = faces.shape[0]  # number of faces



    while faces.shape[0]<16000:
        # print(faces.shape,path)
        N = copy.deepcopy(faces[-1])
        N = N.reshape(1,-1)
        faces = np.concatenate((faces,N),axis=0)
        n_face+=1
        # print(path,)

    # print(path)

    xyz = points[:, :3]# coordinate of vertex shape=[N, 3]
    # normals = points[:,3:6]
    components = path.split("/")
    new_path = "/".join(components[:-2])
    plyname = components[-1]

    origin_path = os.path.join(new_path,"origin")

    origin_path = os.path.join(origin_path,plyname)
    origin_data = PlyData.read(origin_path)
    # print(new_path)
    origin_points = np.array(pd.DataFrame(origin_data.elements[0].data))
    # print('origin_points', origin_points.shape) #(8400, 9)

    if origin_points.shape[1] == 6:
        color = origin_points[:, 3:6]/255  # color of vertex shape=[N, 3]
    else:
        color = origin_points[:, 6:9]/255
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)


    label_face = np.zeros([n_face,1]).astype('int32')
    label_face_onehot = np.zeros([n_face,2]).astype(('int32'))
    """ index of faces shape=[N, 3] """

    # print('faces', faces[0]) #[array([2, 0, 1], dtype=uint32)],每个面的顶点索引
    index_face = np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3).astype('int32')
    # if color.shape[0] < np.max(index_face):
    #     print(path,color.shape[0],np.max(index_face))

    """ RGB of faces of label shape=[N, 3] """
    RGB_face = np.zeros((n_face, 3))
    for i in range(0, n_face):
        # Get the indices of the vertices for this face
        indices = faces[i][0]
        # Get the colors of the vertices
        if points.shape[1] == 6:
            colors = points[indices, 3:6]
        else:
            colors = points[indices, 6:9]

        min_color = np.min(colors, axis=0)
        # Store the result as the face color
        # if min_color[0] == 0:
        #     RGB_face[i] = [0.0,0.0,0.0]
        # else:
        #     RGB_face[i] = [1.0,1.0,1.0]
        # if (min_color==[255,255,255]).all(axis=0):
        #     RGB_face[i] = [1.0,1.0,1.0]
        # else:
        #     RGB_face[i] = [0.0,0.0,0.0]
        RGB_face[i] = min_color

        # print(min_color)




    # Check if indices are within bounds
    max_vertex_index = len(xyz) - 1
    max_normal_index = len(normals) - 1
    max_color_index = len(color) - 1

    # Check for out-of-bounds indices
    max_face_index = np.max(index_face)
    if max_face_index > max_vertex_index or max_face_index > max_normal_index or max_face_index > max_color_index:
        print(f"警告: 文件 {path} 中索引超出范围")
        print(f"  面片最大索引: {max_face_index}")
        print(f"  顶点数组大小: {max_vertex_index + 1}")
        print(f"  法向量数组大小: {max_normal_index + 1}")
        print(f"  颜色数组大小: {max_color_index + 1}")

        # Filter out invalid indices
        valid_mask = (index_face[:, 0] <= max_vertex_index) & (index_face[:, 1] <= max_vertex_index) & (index_face[:, 2] <= max_vertex_index)
        valid_mask &= (index_face[:, 0] <= max_normal_index) & (index_face[:, 1] <= max_normal_index) & (index_face[:, 2] <= max_normal_index)
        valid_mask &= (index_face[:, 0] <= max_color_index) & (index_face[:, 1] <= max_color_index) & (index_face[:, 2] <= max_color_index)

        # Keep only valid faces
        n_valid_faces = np.sum(valid_mask)
        if n_valid_faces == 0:
            print(f"错误: 文件 {path} 没有有效的面片")
            return None, None, None, None, None

        print(f"保留 {n_valid_faces}/{n_face} 个有效面片")
        index_face = index_face[valid_mask]
        faces = faces[valid_mask]
        n_face = n_valid_faces

        # Recalculate arrays with filtered faces
        label_face = np.zeros([n_face, 1]).astype('int32')
        label_face_onehot = np.zeros([n_face, 2]).astype('int32')
        RGB_face = np.zeros((n_face, 3))

    try:
        """ coordinate of 3 vertexes  shape=[N, 9] """
        xyz_face = np.concatenate((xyz[index_face[:, 0], :], xyz[index_face[:, 1], :],xyz[index_face[:, 2], :]), axis=1)
        """  color of 3 vertexes  shape=[N, 9] """
        normal_vertex = np.concatenate((normals[index_face[:, 0], :], normals[index_face[:, 1], :],normals[index_face[:, 2], :]), axis=1)
        color_vertex = np.concatenate((color[index_face[:, 0], :], color[index_face[:, 1], :],color[index_face[:, 2], :]), axis=1)
    except Exception as e:
        print(f"错误: 文件 {path} 在处理面片数据时出错: {e}")
        return None, None, None, None, None



    x1, y1, z1 = xyz_face[:, 0], xyz_face[:, 1], xyz_face[:, 2]
    x2, y2, z2 = xyz_face[:, 3], xyz_face[:, 4], xyz_face[:, 5]
    x3, y3, z3 = xyz_face[:, 6], xyz_face[:, 7], xyz_face[:, 8]
    x_centre = (x1 + x2 + x3) / 3
    y_centre = (y1 + y2 + y3) / 3
    z_centre = (z1 + z2 + z3) / 3
    centre_face = np.concatenate((x_centre.reshape(n_face,1),y_centre.reshape(n_face,1),z_centre.reshape(n_face,1)), axis=1)



    RGB = np.zeros([n_face,3])

    for i in range(0,n_face):

        id0 = faces[i][0][0]
        id1 = faces[i][0][1]
        id2 = faces[i][0][2]
        RGB[i] = [ min(color[id0][0] , color[id1][0] , color[id2][0]),
                  min(color[id0][1] , color[id1][1] , color[id2][1]),
                  min(color[id0][2] , color[id1][2] , color[id2][2])]


    points_face = np.concatenate((xyz_face, centre_face, normal_vertex, RGB, color_vertex), axis=1).astype('float32')
    # points_face = np.concatenate((xyz_face, centre_face, color_vertex, RGB, normal_vertex), axis=1).astype('float32')


    """ get label of each face """
    label_face[(RGB_face == labels).all(axis=1)] = 1
    label_face_onehot[(RGB_face == labels).all(axis=1),1] = 1
    label_face_onehot[(RGB_face != labels).all(axis=1),0] = 1

    # label_face_fake = np.random.randint(0, 2, size=(65536, 1))
    # print(label_face_fake.shape,label_face.shape)
    # print(label_face.sum())

    return index_face, points_face, label_face, label_face_onehot, origin_points

import open3d as o3d

def save_face_predictions(index_face, point_face, label_face, path=" "):
    """
    Input:
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
        label_face: label of face [N, 1]
        path: path to save new generated ply file
    Return:
    """
    # print(type(index_face),type(label_face))

    labels_change_color = [[1,1,1],[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
    name = path[-10:]
    pred_face_path = os.path.join(path[:-10],"pred_face")
    index_face_path = os.path.join(path[:-10],"index_face")
    if not os.path.exists(pred_face_path):
        # os.mkdir(pred_face_path)
        # os.mkdir(index_face_path)
        os.makedirs(pred_face_path, exist_ok=True)
        os.makedirs(index_face_path, exist_ok=True)
    np.save(os.path.join(pred_face_path,name[:-4]+".npy"), label_face.numpy())
    np.save(os.path.join(index_face_path,name[:-4]+".npy"), index_face)

    # input_path = os.path.join(input_path, name)
    # # print(input_path)
    # mesh = o3d.io.read_triangle_mesh(input_path)
    # c = np.asarray(mesh.vertex_colors)

    # for i, l in enumerate(label_face):
    #     index = index_face[i]
    #     if l == 0:
    #         c[index] = labels_change_color[0]
    # for i, l in enumerate(label_face):
    #     index = index_face[i]
    #     if l == 1:
    #         c[index] = labels_change_color[1]

    # mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    # o3d.io.write_triangle_mesh(path, mesh)



class PlyDataset(Dataset):

    def __init__(self, path="data/train", enable_augmentation=False, flip_prob=0.5):
        self.root_path = path
        self.file_list = [f for f in os.listdir(path) if f.endswith('.ply')]

        # 初始化数据增强器
        self.enable_augmentation = enable_augmentation
        if enable_augmentation:
            self.augmenter = DataAugmentation(
                enable_flip=True,
                flip_prob=flip_prob,
                enable_rotation=False,  # 暂时不启用旋转，先专注翻转
                rotation_range=(-10, 10)
            )
            print(f"数据增强已启用: 翻转概率={flip_prob}")
        else:
            self.augmenter = None
            print("数据增强已禁用")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_face, label_face, label_face_onehot, points = load_ply_features(path=read_path)
        raw_points_face = points_face.copy()

        # === 数据增强 ===
        if self.enable_augmentation and self.augmenter is not None:
            # 执行数据增强
            points_face = self.augmenter.augment(points_face, is_training=True)

        # breakpoint()
        # print('points_face', points_face.shape) #(16000, 33) xyz_face：9 维（3 个顶点的坐标）。centre_face：3 维（中心点坐标）。normal_vertex：9 维（3 个顶点的法向量）。RGB：3 维（面的颜色）。color_vertex：9 维（3 个顶点的颜色）。
        # print('label_face', label_face.shape) #(16000, 1)
        # print('index_face', index_face.shape) #(16000, 3)
        # print('label_face_onehot_', label_face_onehot[0]) #[1 0]
        # print('label_face_', label_face[0]) #[0]
        # print('label_face_onehot', label_face_onehot.shape) #(16000,2)
        # print('points', points.shape) #原始点云数据


        # move all mesh to origin
        centre = points_face[:, 9:12].mean(axis=0) #[0:3],[3:6],[6:9]是顶点，[9:12]是中心点
        points_face[:, 0:3] -= centre
        points_face[:, 3:6] -= centre
        points_face[:, 6:9] -= centre
        points_face[:, 9:12] = (points_face[:, 0:3] + points_face[:, 3:6] + points_face[:, 6:9]) / 3
        points[:, :3] -= centre
        max = points.max()
        points_face[:, :12] = points_face[:, :12] / max

        # normalized data
        maxs = points[:, :3].max(axis=0)
        mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = points[:, :3].std(axis=0)
        nmeans = points[:, 3:].mean(axis=0)
        nstds = points[:, 3:].std(axis=0)
        nmeans_f = points_face[:, 21:24].mean(axis=0)
        nstds_f = points_face[:, 21:24].std(axis=0)
        for i in range(3):
            #normalize coordinate
            points_face[:, i] = (points_face[:, i] - means[i]) / stds[i]  # point 1
            points_face[:, i + 3] = (points_face[:, i + 3] - means[i]) / stds[i]  # point 2
            points_face[:, i + 6] = (points_face[:, i + 6] - means[i]) / stds[i]  # point 3
            points_face[:, i + 9] = (points_face[:, i + 9] - mins[i]) / (maxs[i] - mins[i])  # centre
        # #     #normalize normal vector
            # points_face[:, i + 12] = (points_face[:, i + 12] - nmeans[i]) / nstds[i]  # normal1
            # points_face[:, i + 15] = (points_face[:, i + 15] - nmeans[i]) / nstds[i]  # normal2
            # points_face[:, i + 18] = (points_face[:, i + 18] - nmeans[i]) / nstds[i]  # normal3
            # points_face[:, i + 21] = (points_face[:, i + 21] - nmeans_f[i]) / nstds_f[i]  # face normal


        return index_face, points_face, label_face, label_face_onehot, self.file_list[item], raw_points_face




# Backward-compatible aliases for historical checkpoints and scripts.  New
# code should use the PEP 8 names above.
get_data = load_ply_features
generate_plyfile = save_face_predictions
plydataset = PlyDataset
