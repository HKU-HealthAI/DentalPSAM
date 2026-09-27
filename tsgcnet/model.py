import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
from collections import defaultdict, deque
from .utils import gather_adjacent_features


def find_adjacent_faces(index_face):
    """
    找到每个三角面的邻接面（共享至少一个顶点的面）

    参数:
        index_face: numpy数组，形状为[N,3]，表示N个三角面的顶点索引

    返回:
        adjacent_faces: 列表的列表，adjacent_faces[i]是第i个面的所有邻接面索引
    """
    N = index_face.shape[0]  # 面数（16000）

    # 1. 构建顶点到面的映射字典：{顶点索引: [包含该顶点的面索引列表]}
    vertex_to_faces = defaultdict(list)
    for face_idx in range(N):
        for vertex in index_face[face_idx]:
            vertex_to_faces[int(vertex)].append(int(face_idx))

    # print('vertex_to_faces', vertex_to_faces[0])

    # 2. 为每个面查找邻接面
    adjacent_faces = []
    for face_idx in range(N):
        # 获取当前面的三个顶点
        vertices = index_face[face_idx]

        # 收集这三个顶点关联的所有面
        neighbor_faces = set()
        for vertex in vertices:
            for neighbor_face in vertex_to_faces[int(vertex)]:
                if int(neighbor_face) != int(face_idx):  # 排除自己
                    neighbor_faces.add(int(neighbor_face))

        adjacent_faces.append(list(neighbor_faces))

    # print('adjacent_faces', adjacent_faces)

    return adjacent_faces

def find_adjacent_faces_extended(index_face, min_neighbors=12):
    """
    找到每个三角面的邻接面（共享至少一个顶点的面），如果邻接面不足min_neighbors个，
    则继续查找邻接面的邻接面，直到满足数量要求

    参数:
        index_face: numpy数组，形状为[N,3]，表示N个三角面的顶点索引
        min_neighbors: 每个面需要找到的最少邻接面数量，默认为10

    返回:
        adjacent_faces: 列表的列表，adjacent_faces[i]是第i个面的所有邻接面索引
                       如果直接邻接面不足min_neighbors，会包含更远的邻接面
    """
    N = index_face.shape[0]  # 面数

    # 1. 构建顶点到面的映射字典
    vertex_to_faces = defaultdict(list)
    for face_idx in range(N):
        for vertex in index_face[face_idx]:
            vertex_to_faces[int(vertex)].append(int(face_idx))

    # 2. 先找到直接邻接面（原始方法）
    direct_adjacent = []
    for face_idx in range(N):
        vertices = index_face[face_idx]
        neighbor_faces = set()
        for vertex in vertices:
            for neighbor_face in vertex_to_faces[int(vertex)]:
                if int(neighbor_face) != int(face_idx):
                    neighbor_faces.add(int(neighbor_face))
        direct_adjacent.append(neighbor_faces)

    # 3. 对于邻接面不足min_neighbors的面，扩展搜索范围
    adjacent_faces = []
    for face_idx in range(N):
        if len(direct_adjacent[face_idx]) >= min_neighbors:
            # 如果直接邻接面已经足够，直接使用
            adjacent_faces.append(list(direct_adjacent[face_idx]))
            continue

        # 使用广度优先搜索(BFS)扩展查找
        visited = set([face_idx])
        queue = deque(direct_adjacent[face_idx])
        extended_neighbors = set(direct_adjacent[face_idx])

        while len(extended_neighbors) < min_neighbors and queue:
            current_face = queue.popleft()

            # 添加当前面的邻接面（排除已经访问过的）
            for neighbor in direct_adjacent[current_face]:
                if neighbor not in visited and neighbor != face_idx:
                    visited.add(neighbor)
                    extended_neighbors.add(neighbor)
                    queue.append(neighbor)

                    # 如果已经找到足够的邻接面，可以提前退出
                    if len(extended_neighbors) >= min_neighbors:
                        break

        # 转换为列表并截取前min_neighbors个（如果需要排序可以在这里添加）
        neighbors_list = list(extended_neighbors)
        if len(neighbors_list) > min_neighbors:
            neighbors_list = neighbors_list[:min_neighbors]

        adjacent_faces.append(neighbors_list)

    return adjacent_faces

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]  # (batch_size, num_points, k)
    return idx
# def knn(x, k):

#     pairwise_distance = torch.cdist(x.transpose(2, 1), x.transpose(2, 1))  # (batch_size, num_points, num_points)
#     idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]  # (batch_size, num_points, k)
#     return idx

#提取指定位置的点
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def get_graph_feature(coor, nor,cor , k=10):
    batch_size, num_dims, num_points  = coor.shape
    coor = coor.view(batch_size, -1, num_points)
    # print('coor.shape', coor.shape) #[1, 9, 16000]->[1, 64, 16000]->[1, 128, 16000]
    # print('nor.shape', nor.shape) #同上
    # print('cor.shape', cor.shape) #同上

    idx = knn(coor, k=k)
    # print('idx', idx.shape) #torch.Size([1, 16000, 12])
    index = idx
    device = coor.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = coor.size()
    _, num_dims2, _ = nor.size()
    _, num_dims3, _ = cor.size()

    coor = coor.transpose(2,1).contiguous()
    nor = nor.transpose(2,1).contiguous()
    cor = cor.transpose(2,1).contiguous()

    # coordinate
    coor_feature = coor.view(batch_size * num_points, -1)[idx, :]
    # print('coor_feature', coor_feature.shape) #[192000, 9]
    coor_feature = coor_feature.view(batch_size, num_points, k, num_dims)
    # print('coor_feature', coor_feature.shape) #[1, 16000, 12, 9]
    coor = coor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # print('coor', coor.shape) #[1, 16000, 12, 9]
    coor_feature = torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()

    # normal vector
    nor_feature = nor.view(batch_size * num_points, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, num_points, k, num_dims2)
    nor = nor.view(batch_size, num_points, 1, num_dims2).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()

    # color vector
    cor_feature = cor.view(batch_size * num_points, -1)[idx, :]
    cor_feature = cor_feature.view(batch_size, num_points, k, num_dims3)
    cor = cor.view(batch_size, num_points, 1, num_dims3).repeat(1, 1, k, 1)
    cor_feature = torch.cat((cor_feature, cor), dim=3).permute(0, 3, 1, 2).contiguous()
    # print('coor_feature', coor_feature.shape) #[1, 18, 16000, 12]->[1, 128, 16000, 12]->[1, 256, 16000, 12]
    # print('nor_feature', nor_feature.shape) #同上
    # print('cor_feature', cor_feature.shape) #同上
    return coor_feature, nor_feature, cor_feature, index



class GraphAttention(nn.Module):
    def __init__(self,feature_dim,out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.K=K

    def forward(self, Graph_index, x, feature):

        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0,2,3,1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat([centre-neighbor_feature, neighbor_feature], dim=3).permute(0,3,2,1)
        e = self.conv(delta_f)
        e = e.permute(0,3,2,1)
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        graph_feature = torch.sum(torch.mul(attention, feature),dim = 2) .permute(0,2,1)
        return graph_feature


class TSGCNet(nn.Module):
    def __init__(self, k=16, in_channels=12, output_channels=8):
        super(TSGCNet, self).__init__()
        self.k = k
        ''' coordinate stream '''
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels*2, 64, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv2_c = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn2_c,
                                   nn.LeakyReLU(negative_slope=0.2))



        self.conv3_c = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn3_c,
                                   nn.LeakyReLU(negative_slope=0.2))



        self.conv4_c = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.attention_layer1_c = GraphAttention(feature_dim=9, out_dim=64, K=self.k)
        self.attention_layer2_c = GraphAttention(feature_dim=64, out_dim=128, K=self.k)
        self.attention_layer3_c = GraphAttention(feature_dim=128, out_dim=256, K=self.k)
        self.FTM_c1 = STNkd(k=9)
        ''' normal stream '''
        self.bn1_n = nn.BatchNorm2d(64)
        self.bn2_n = nn.BatchNorm2d(128)
        self.bn3_n = nn.BatchNorm2d(256)
        self.bn4_n = nn.BatchNorm1d(512)
        self.conv1_n = nn.Sequential(nn.Conv2d((in_channels)*2, 64, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv2_n = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv3_n = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))



        self.conv4_n = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.FTM_n1 = STNkd(k=9)

        # color stream
        self.bn1_cor = nn.BatchNorm2d(64)
        self.bn2_cor = nn.BatchNorm2d(128)
        self.bn3_cor = nn.BatchNorm2d(256)
        self.bn4_cor = nn.BatchNorm1d(512)
        self.conv1_cor = nn.Sequential(nn.Conv2d((in_channels)*2, 64, kernel_size=1, bias=False),
                                     self.bn1_cor,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv2_cor = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                     self.bn2_cor,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv3_cor = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                     self.bn3_cor,
                                     nn.LeakyReLU(negative_slope=0.2))



        self.conv4_cor = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_cor,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.FTM_cor1 = STNkd(k=9)


        '''feature-wise attention'''

        self.fa = nn.Sequential(nn.Conv1d(1024+512, 1024+512, kernel_size=1, bias=False),
                                nn.BatchNorm1d(1024+512),
                                nn.LeakyReLU(0.2))

        # self.fa = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
        #                         nn.BatchNorm1d(1024),
        #                         nn.LeakyReLU(0.2))
        ''' feature fusion '''
        self.pred1 = nn.Sequential(nn.Conv1d(1024+512, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.pred1 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(512),
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.pred2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))
        self.dp1 = nn.Dropout(p=0.6)
        self.dp2 = nn.Dropout(p=0.6)
        self.dp3 = nn.Dropout(p=0.6)
        # Mesh connectivity does not change across epochs. Cache the CPU index
        # tensor per face array and transfer it only when a forward pass needs it.
        self._adjacency_cache = {}

    def _adjacency_index(self, index_face):
        """Build or reuse the fixed-width face-neighbour index for one mesh."""
        faces = np.ascontiguousarray(index_face[0])
        cache_key = faces.tobytes()
        processed_adjacent = self._adjacency_cache.get(cache_key)
        if processed_adjacent is None:
            adjacent_faces = find_adjacent_faces_extended(faces)
            processed_adjacent = torch.zeros((len(faces), self.k), dtype=torch.long)
            for face_index, neighbours in enumerate(adjacent_faces):
                if not neighbours:
                    processed_adjacent[face_index] = face_index
                elif len(neighbours) >= self.k:
                    processed_adjacent[face_index] = torch.tensor(neighbours[:self.k], dtype=torch.long)
                else:
                    processed_adjacent[face_index, :len(neighbours)] = torch.tensor(neighbours, dtype=torch.long)
                    processed_adjacent[face_index, len(neighbours):] = neighbours[0]
            self._adjacency_cache[cache_key] = processed_adjacent
        return processed_adjacent

    def forward(self, x, index_face):
        coor = x[:, :9,:]
        nor = x[:, 12:21,:]
        cor = x[:,24:33,:]

        trans_c = self.FTM_c1(coor)
        coor = coor.transpose(2, 1)
        coor = torch.bmm(coor, trans_c)
        coor = coor.transpose(2, 1)

        trans_n = self.FTM_n1(nor)
        nor = nor.transpose(2, 1)
        nor = torch.bmm(nor, trans_n)
        nor = nor.transpose(2, 1)

        trans_cor = self.FTM_cor1(cor)
        cor = cor.transpose(2, 1)
        cor = torch.bmm(cor, trans_cor)
        cor = cor.transpose(2, 1)
        processed_adjacent = self._adjacency_index(index_face)
        index = processed_adjacent.unsqueeze(0).to(coor.device)
        # print('processed_adjacent', processed_adjacent[13])

        # coor1_aj = gather_adjacent_features(coor, processed_adjacent)
        # print('coor1_aj', coor1_aj.shape)

        # adjacent_faces = find_adjacent_faces(index_face[0])
        # def get_adjacent_length_stats_with_indices(adjacent_faces):
        #     """
        #     计算邻接面列表的长度统计信息，并返回最长/最短邻接面的索引

        #     参数:
        #         adjacent_faces: 二维列表，adjacent_faces[i]是第i个面的邻接面索引列表

        #     返回:
        #         (max_len, max_indices, min_len, min_indices):
        #             - max_len: 最长邻接面列表的长度
        #             - max_indices: 所有具有最长邻接面的面索引列表
        #             - min_len: 最短邻接面列表的长度
        #             - min_indices: 所有具有最短邻接面的面索引列表
        #     """
        #     # 获取所有邻接面列表的长度
        #     lengths = [len(adj_list) for adj_list in adjacent_faces]

        #     max_len = max(lengths)
        #     min_len = min(lengths)

        #     # 找到所有具有最长/最短邻接面的面索引
        #     max_indices = [i for i, length in enumerate(lengths) if length == max_len]
        #     min_indices = [i for i, length in enumerate(lengths) if length == min_len]

        #     return max_len, max_indices, min_len, min_indices
        # max_len, max_indices, min_len, min_indices = get_adjacent_length_stats_with_indices(adjacent_faces)
        # # print(f"最长邻接面长度: {max_len}, 对应的面索引: {max_indices}")  # 输出: 最长邻接面长度: 4, 对应的面索引: [2]
        # print(f"最短邻接面长度: {min_len}")
        # 打印结果
        # for min_indice in min_indices:
        #     print('min_indice', adjacent_faces[min_indice])




        # print('')

        # coor1, nor1,cor1 ,index = get_graph_feature(coor, nor,cor, k=self.k)
        # print('index', index.shape)
        coor1 = gather_adjacent_features(coor, processed_adjacent)
        nor1 = gather_adjacent_features(nor, processed_adjacent)
        cor1 = gather_adjacent_features(cor, processed_adjacent)
        # print('coor1_0', coor1.shape) #[1, 18, 16000, 12]
        coor1 = self.conv1_c(coor1)
        # print('coor1_1', coor1.shape) #[1, 64, 16000, 12]
        nor1 = self.conv1_n(nor1)
        coor1 = self.attention_layer1_c(index, coor, coor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]
        # print('coor1', coor1.shape) #[1, 64, 16000]
        # print('nor1', nor1.shape) #[1, 64, 16000]



        cor1 = self.conv1_cor(cor1)
        cor1 = cor1.max(dim=-1, keepdim=False)[0]

        # coor2, nor2,cor2, index = get_graph_feature(coor1, nor1,cor1, k=self.k)
        coor2 = gather_adjacent_features(coor1, processed_adjacent)
        nor2 = gather_adjacent_features(nor1, processed_adjacent)
        cor2 = gather_adjacent_features(cor1, processed_adjacent)
        # print('coor2', coor2.shape) #[1, 128, 16000, 12]
        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.attention_layer2_c(index, coor1, coor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]

        cor2 = self.conv2_cor(cor2)
        cor2 = cor2.max(dim=-1, keepdim=False)[0]

        # coor3, nor3,cor3, index = get_graph_feature(coor2, nor2,cor2, k=self.k)
        coor3 = gather_adjacent_features(coor2, processed_adjacent)
        nor3 = gather_adjacent_features(nor2, processed_adjacent)
        cor3 = gather_adjacent_features(cor2, processed_adjacent)

        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.attention_layer3_c(index, coor2, coor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]

        cor3 = self.conv3_cor(cor3)
        cor3 = cor3.max(dim=-1, keepdim=False)[0]

        coor = torch.cat((coor1, coor2,coor3), dim=1)
        coor = self.conv4_c(coor)
        nor = torch.cat((nor1, nor2,nor3), dim=1)
        nor = self.conv4_n(nor)
        cor = torch.cat((cor1, cor2,cor3), dim=1)
        cor = self.conv4_cor(cor)

        avgSum_coor = coor.sum(1)/512
        avgSum_nor = nor.sum(1)/512
        avgSum_cor = cor.sum(1)/512
        avgSum = avgSum_coor+avgSum_nor+avgSum_cor
        # avgSum = avgSum_cor+avgSum_nor
        weight_coor = (avgSum_coor / avgSum).reshape(1,1,16000)
        weight_nor = (avgSum_nor / avgSum).reshape(1,1,16000)
        weight_cor = (avgSum_cor / avgSum).reshape(1,1,16000)
        # print('weight_coor',  weight_coor)
        # print('weight_nor',  weight_nor)
        # print('weight_cor',  weight_cor)
        x = torch.cat((coor*weight_coor, nor*weight_nor, cor*weight_cor), dim=1)
        # x = torch.cat((coor*weight_nor, nor*weight_cor), dim=1)
        # x = torch.cat((coor*weight_coor,  nor*weight_nor), dim=1)

        weight = self.fa(x)
        # print('weight', weight.shape) #[1, 1536, 16000]
        x = weight*x

        x = self.pred1(x)
        self.dp1(x)
        x = self.pred2(x)
        self.dp2(x)
        x = self.pred3(x)
        self.dp3(x)
        score = self.pred4(x)
        score = F.log_softmax(score, dim=1)
        score = score.permute(0, 2, 1)

        return score
