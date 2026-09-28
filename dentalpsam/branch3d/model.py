"""Color-aware TSGCNet with the native checkpoint parameter names preserved.

Forward inputs are [1, 33, 16000] float features on the model device and
[1, 16000, 3] CPU face indices. The output is [1, 16000, classes] log
probabilities. Adjacency construction is CPU-based; inference uses batch one.
"""

from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import gather_adjacent_features


def find_adjacent_faces(index_face):
    """Return faces sharing at least one vertex with each [N, 3] input face."""
    N = index_face.shape[0]

    vertex_to_faces = defaultdict(list)
    for face_idx in range(N):
        for vertex in index_face[face_idx]:
            vertex_to_faces[int(vertex)].append(int(face_idx))

    adjacent_faces = []
    for face_idx in range(N):

        vertices = index_face[face_idx]

        neighbor_faces = set()
        for vertex in vertices:
            for neighbor_face in vertex_to_faces[int(vertex)]:
                if int(neighbor_face) != int(face_idx):
                    neighbor_faces.add(int(neighbor_face))

        adjacent_faces.append(list(neighbor_faces))

    return adjacent_faces


def find_adjacent_faces_extended(index_face, min_neighbors=12):
    """Expand vertex-sharing neighbors by breadth-first search when necessary.

    Preserve native set iteration order: sorting would change the truncated
    neighborhood seen by existing checkpoints. Disconnected faces can return
    fewer than min_neighbors; the caller pads to the network's fixed width.
    """
    N = index_face.shape[0]

    vertex_to_faces = defaultdict(list)
    for face_idx in range(N):
        for vertex in index_face[face_idx]:
            vertex_to_faces[int(vertex)].append(int(face_idx))

    direct_adjacent = []
    for face_idx in range(N):
        vertices = index_face[face_idx]
        neighbor_faces = set()
        for vertex in vertices:
            for neighbor_face in vertex_to_faces[int(vertex)]:
                if int(neighbor_face) != int(face_idx):
                    neighbor_faces.add(int(neighbor_face))
        direct_adjacent.append(neighbor_faces)

    adjacent_faces = []
    for face_idx in range(N):
        if len(direct_adjacent[face_idx]) >= min_neighbors:

            adjacent_faces.append(list(direct_adjacent[face_idx]))
            continue

        visited = set([face_idx])
        queue = deque(direct_adjacent[face_idx])
        extended_neighbors = set(direct_adjacent[face_idx])

        while len(extended_neighbors) < min_neighbors and queue:
            current_face = queue.popleft()

            for neighbor in direct_adjacent[current_face]:
                if neighbor not in visited and neighbor != face_idx:
                    visited.add(neighbor)
                    extended_neighbors.add(neighbor)
                    queue.append(neighbor)

                    if len(extended_neighbors) >= min_neighbors:
                        break

        neighbors_list = list(extended_neighbors)
        if len(neighbors_list) > min_neighbors:
            neighbors_list = neighbors_list[:min_neighbors]

        adjacent_faces.append(neighbors_list)

    return adjacent_faces


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]
    return idx


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
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


class STNkd(nn.Module):
    """Learn a k-by-k feature transform with checkpoint-compatible layer names."""

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

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def get_graph_feature(coor, nor, cor, k=10):
    batch_size, num_dims, num_points = coor.shape
    coor = coor.view(batch_size, -1, num_points)

    idx = knn(coor, k=k)

    index = idx
    device = coor.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = coor.size()
    _, num_dims2, _ = nor.size()
    _, num_dims3, _ = cor.size()

    coor = coor.transpose(2, 1).contiguous()
    nor = nor.transpose(2, 1).contiguous()
    cor = cor.transpose(2, 1).contiguous()

    coor_feature = coor.view(batch_size * num_points, -1)[idx, :]

    coor_feature = coor_feature.view(batch_size, num_points, k, num_dims)

    coor = coor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    coor_feature = (
        torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()
    )

    nor_feature = nor.view(batch_size * num_points, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, num_points, k, num_dims2)
    nor = nor.view(batch_size, num_points, 1, num_dims2).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()

    cor_feature = cor.view(batch_size * num_points, -1)[idx, :]
    cor_feature = cor_feature.view(batch_size, num_points, k, num_dims3)
    cor = cor.view(batch_size, num_points, 1, num_dims3).repeat(1, 1, k, 1)
    cor_feature = torch.cat((cor_feature, cor), dim=3).permute(0, 3, 1, 2).contiguous()

    return coor_feature, nor_feature, cor_feature, index


class GraphAttention(nn.Module):
    """Aggregate coordinate neighbors using channel-wise attention."""

    def __init__(self, feature_dim, out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.K = K

    def forward(self, Graph_index, x, feature):

        B, C, N = x.shape
        # Preserve the native reshape, not a transpose: layout affects weights.
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0, 2, 3, 1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat(
            [centre - neighbor_feature, neighbor_feature], dim=3
        ).permute(0, 3, 2, 1)
        e = self.conv(delta_f)
        e = e.permute(0, 3, 2, 1)
        attention = F.softmax(e, dim=2)
        graph_feature = torch.sum(torch.mul(attention, feature), dim=2).permute(0, 2, 1)
        return graph_feature


class TSGCNet(nn.Module):
    """Coordinate, normal, and color streams for per-triangle classification.

    Plaque entry points use in_channels=9, output_channels=2, and k=12.
    Registered layer names and initialization order match existing checkpoints.
    """

    def __init__(self, k=16, in_channels=12, output_channels=8, normalization="per_mesh"):
        super(TSGCNet, self).__init__()
        if normalization not in ("per_mesh", "running"):
            raise ValueError("Unknown 3D branch normalization")
        self.normalization = normalization
        self.k = k
        # Coordinate stream uses graph attention after neighborhood convolutions.
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
            self.bn1_c,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2_c = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn2_c,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3_c = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn3_c,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv4_c = nn.Sequential(
            nn.Conv1d(448, 512, kernel_size=1, bias=False),
            self.bn4_c,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.attention_layer1_c = GraphAttention(feature_dim=9, out_dim=64, K=self.k)
        self.attention_layer2_c = GraphAttention(feature_dim=64, out_dim=128, K=self.k)
        self.attention_layer3_c = GraphAttention(feature_dim=128, out_dim=256, K=self.k)
        self.FTM_c1 = STNkd(k=9)
        # Normal stream uses neighbor max pooling.
        self.bn1_n = nn.BatchNorm2d(64)
        self.bn2_n = nn.BatchNorm2d(128)
        self.bn3_n = nn.BatchNorm2d(256)
        self.bn4_n = nn.BatchNorm1d(512)
        self.conv1_n = nn.Sequential(
            nn.Conv2d((in_channels) * 2, 64, kernel_size=1, bias=False),
            self.bn1_n,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2_n = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn2_n,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3_n = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn3_n,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv4_n = nn.Sequential(
            nn.Conv1d(448, 512, kernel_size=1, bias=False),
            self.bn4_n,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.FTM_n1 = STNkd(k=9)
        # Color stream consumes origin RGB, not annotation colors.
        self.bn1_cor = nn.BatchNorm2d(64)
        self.bn2_cor = nn.BatchNorm2d(128)
        self.bn3_cor = nn.BatchNorm2d(256)
        self.bn4_cor = nn.BatchNorm1d(512)
        self.conv1_cor = nn.Sequential(
            nn.Conv2d((in_channels) * 2, 64, kernel_size=1, bias=False),
            self.bn1_cor,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2_cor = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn2_cor,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3_cor = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn3_cor,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv4_cor = nn.Sequential(
            nn.Conv1d(448, 512, kernel_size=1, bias=False),
            self.bn4_cor,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.FTM_cor1 = STNkd(k=9)

        self.fa = nn.Sequential(
            nn.Conv1d(1024 + 512, 1024 + 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024 + 512),
            nn.LeakyReLU(0.2),
        )

        self.pred1 = nn.Sequential(
            nn.Conv1d(1024 + 512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.pred2 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pred3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.pred4 = nn.Sequential(
            nn.Conv1d(128, output_channels, kernel_size=1, bias=False)
        )
        self.dp1 = nn.Dropout(p=0.6)
        self.dp2 = nn.Dropout(p=0.6)
        self.dp3 = nn.Dropout(p=0.6)

        # Cache CPU connectivity; transfer indices only when needed for inference.
        self._adjacency_cache = {}
        self.train(self.training)

    def train(self, mode=True):
        """Use the current mesh's BN moments in both training and inference.

        One mesh is processed at a time. Running buffers remain registered for
        strict checkpoint compatibility, but new models neither read nor update
        them. Dropout and every non-BN module still follow the requested mode.
        The running-statistics path is only for loading older checkpoints.
        """
        super().train(mode)
        if self.normalization == "per_mesh":
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.track_running_stats = False
                    module.train(True)
        return self

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
                    processed_adjacent[face_index] = torch.tensor(
                        neighbours[: self.k], dtype=torch.long
                    )
                else:
                    processed_adjacent[face_index, : len(neighbours)] = torch.tensor(
                        neighbours, dtype=torch.long
                    )
                    processed_adjacent[face_index, len(neighbours) :] = neighbours[0]
            self._adjacency_cache[cache_key] = processed_adjacent
        return processed_adjacent

    def forward(self, x, index_face):
        """Map [1, 33, 16000] features to [1, 16000, classes] log probabilities."""
        coor = x[:, :9, :]
        nor = x[:, 12:21, :]
        cor = x[:, 24:33, :]

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

        coor1 = gather_adjacent_features(coor, processed_adjacent)
        nor1 = gather_adjacent_features(nor, processed_adjacent)
        cor1 = gather_adjacent_features(cor, processed_adjacent)

        coor1 = self.conv1_c(coor1)

        nor1 = self.conv1_n(nor1)
        coor1 = self.attention_layer1_c(index, coor, coor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]

        cor1 = self.conv1_cor(cor1)
        cor1 = cor1.max(dim=-1, keepdim=False)[0]

        coor2 = gather_adjacent_features(coor1, processed_adjacent)
        nor2 = gather_adjacent_features(nor1, processed_adjacent)
        cor2 = gather_adjacent_features(cor1, processed_adjacent)

        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.attention_layer2_c(index, coor1, coor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]

        cor2 = self.conv2_cor(cor2)
        cor2 = cor2.max(dim=-1, keepdim=False)[0]

        coor3 = gather_adjacent_features(coor2, processed_adjacent)
        nor3 = gather_adjacent_features(nor2, processed_adjacent)
        cor3 = gather_adjacent_features(cor2, processed_adjacent)

        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.attention_layer3_c(index, coor2, coor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]

        cor3 = self.conv3_cor(cor3)
        cor3 = cor3.max(dim=-1, keepdim=False)[0]

        coor = torch.cat((coor1, coor2, coor3), dim=1)
        coor = self.conv4_c(coor)
        nor = torch.cat((nor1, nor2, nor3), dim=1)
        nor = self.conv4_n(nor)
        cor = torch.cat((cor1, cor2, cor3), dim=1)
        cor = self.conv4_cor(cor)

        avgSum_coor = coor.sum(1) / 512
        avgSum_nor = nor.sum(1) / 512
        avgSum_cor = cor.sum(1) / 512
        avgSum = avgSum_coor + avgSum_nor + avgSum_cor

        weight_coor = (avgSum_coor / avgSum).reshape(1, 1, 16000)
        weight_nor = (avgSum_nor / avgSum).reshape(1, 1, 16000)
        weight_cor = (avgSum_cor / avgSum).reshape(1, 1, 16000)

        x = torch.cat((coor * weight_coor, nor * weight_nor, cor * weight_cor), dim=1)

        weight = self.fa(x)

        x = weight * x

        x = self.pred1(x)
        # Native training discards dropout outputs. Assigning them to x changes
        # the training protocol; preserve these calls for source compatibility.
        self.dp1(x)
        x = self.pred2(x)
        self.dp2(x)
        x = self.pred3(x)
        self.dp3(x)
        score = self.pred4(x)
        score = F.log_softmax(score, dim=1)
        score = score.permute(0, 2, 1)

        return score
