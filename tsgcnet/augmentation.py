"""
数据增强模块 - 专门为牙科3D网格分割设计
垂直翻转增强：模拟上下牙齿的镜像关系
"""

import numpy as np
import random

class DataAugmentation:
    """数据增强类"""

    def __init__(self, enable_flip=True, flip_prob=0.5, enable_rotation=False, rotation_range=(-10, 10), seed=None):
        """
        初始化数据增强

        Args:
            enable_flip: 是否启用垂直翻转
            flip_prob: 翻转概率
            enable_rotation: 是否启用旋转增强
            rotation_range: 旋转角度范围（度）
            seed: 随机种子
        """
        self.enable_flip = enable_flip
        self.flip_prob = flip_prob
        self.enable_rotation = enable_rotation
        self.rotation_range = rotation_range

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def vertical_flip(self, points_face):
        """
        垂直翻转：模拟上下牙齿的镜像关系
        翻转Y轴（上下方向），保持Z轴（前后）和X轴（左右）不变

        Args:
            points_face: 面片特征 [N, 33]

        Returns:
            翻转后的 points_face
        """
        points_face_flipped = points_face.copy()

        # 翻转Y坐标（上下方向）
        # 顶点1: xyz (0:3)
        points_face_flipped[:, 1] *= -1
        # 顶点2: xyz (3:6)
        points_face_flipped[:, 4] *= -1
        # 顶点3: xyz (6:9)
        points_face_flipped[:, 7] *= -1
        # 中心点: centre (9:12)
        points_face_flipped[:, 10] *= -1

        # 翻转法向量Y分量
        # 顶点1法向量: normal1 (12:15) -> Y分量是索引13
        points_face_flipped[:, 13] *= -1
        # 顶点2法向量: normal2 (15:18) -> Y分量是索引16
        points_face_flipped[:, 16] *= -1
        # 顶点3法向量: normal3 (18:21) -> Y分量是索引19
        points_face_flipped[:, 19] *= -1

        # 注意：RGB颜色(21:24)和顶点颜色(24:33)不需要翻转

        return points_face_flipped

    def random_rotation_z(self, points_face, angle_range=None):
        """
        绕Z轴随机旋转（模拟不同拍摄角度）

        Args:
            points_face: 面片特征 [N, 33]
            angle_range: 旋转角度范围，如果为None则使用初始化时的范围

        Returns:
            旋转后的 points_face
        """
        if angle_range is None:
            angle_range = self.rotation_range

        angle = np.random.uniform(angle_range[0], angle_range[1])
        theta = np.radians(angle)

        # Z轴旋转矩阵
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        points_face_rotated = points_face.copy()

        # 旋转顶点坐标
        for i in range(3):  # 3个顶点
            start_idx = i * 3
            end_idx = start_idx + 3
            coords = points_face_rotated[:, start_idx:end_idx]
            points_face_rotated[:, start_idx:end_idx] = coords @ rot_matrix.T

        # 旋转中心点坐标 (9:12)
        centre = points_face_rotated[:, 9:12]
        points_face_rotated[:, 9:12] = centre @ rot_matrix.T

        # 旋转法向量
        for i in range(3):  # 3个法向量
            start_idx = 12 + i * 3
            end_idx = start_idx + 3
            normal = points_face_rotated[:, start_idx:end_idx]
            points_face_rotated[:, start_idx:end_idx] = normal @ rot_matrix.T

        return points_face_rotated

    def augment(self, points_face, is_training=True):
        """
        执行数据增强

        Args:
            points_face: 面片特征 [N, 33]
            is_training: 是否为训练模式

        Returns:
            增强后的 points_face
        """
        if not is_training:
            return points_face

        # 垂直翻转
        if self.enable_flip and np.random.random() < self.flip_prob:
            points_face = self.vertical_flip(points_face)

        # 随机旋转
        if self.enable_rotation and np.random.random() < 0.3:  # 30%概率旋转
            points_face = self.random_rotation_z(points_face)

        return points_face

    def get_augmentation_info(self):
        """
        获取数据增强配置信息

        Returns:
            配置信息字典
        """
        return {
            'enable_flip': self.enable_flip,
            'flip_prob': self.flip_prob,
            'enable_rotation': self.enable_rotation,
            'rotation_range': self.rotation_range
        }

def test_augmentation():
    """
    测试数据增强功能
    """
    print("测试数据增强功能...")

    # 创建测试数据
    test_points = np.array([
        [1.0, 2.0, 3.0,   # 顶点1
         4.0, 5.0, 6.0,   # 顶点2
         7.0, 8.0, 9.0,   # 顶点3
         5.0, 5.0, 5.0,   # 中心点
         0.1, 0.2, 0.3,   # 法向量1
         0.4, 0.5, 0.6,   # 法向量2
         0.7, 0.8, 0.9,   # 法向量3
         0.5, 0.5, 0.5,   # RGB
         0.5, 0.5, 0.5,   # 顶点1颜色
         0.5, 0.5, 0.5,   # 顶点2颜色
         0.5, 0.5, 0.5]   # 顶点3颜色
    ])

    # 创建数据增强器
    augmenter = DataAugmentation(
        enable_flip=True,
        flip_prob=1.0,  # 100%翻转用于测试
        enable_rotation=True,
        rotation_range=(-15, 15)
    )

    print("原始数据:")
    print(f"  顶点1 Y: {test_points[0, 1]:.3f}")
    print(f"  顶点2 Y: {test_points[0, 4]:.3f}")
    print(f"  顶点3 Y: {test_points[0, 7]:.3f}")
    print(f"  中心点 Y: {test_points[0, 10]:.3f}")
    print(f"  法向量1 Y: {test_points[0, 13]:.3f}")
    print(f"  法向量2 Y: {test_points[0, 16]:.3f}")
    print(f"  法向量3 Y: {test_points[0, 19]:.3f}")

    # 测试垂直翻转
    flipped_points = augmenter.vertical_flip(test_points)
    print("\n垂直翻转后:")
    print(f"  顶点1 Y: {flipped_points[0, 1]:.3f}")
    print(f"  顶点2 Y: {flipped_points[0, 4]:.3f}")
    print(f"  顶点3 Y: {flipped_points[0, 7]:.3f}")
    print(f"  中心点 Y: {flipped_points[0, 10]:.3f}")
    print(f"  法向量1 Y: {flipped_points[0, 13]:.3f}")
    print(f"  法向量2 Y: {flipped_points[0, 16]:.3f}")
    print(f"  法向量3 Y: {flipped_points[0, 19]:.3f}")

    # 验证翻转效果
    expected_y = -test_points[0, 1]
    actual_y = flipped_points[0, 1]
    print(f"\n翻转验证: 期望 {expected_y:.3f}, 实际 {actual_y:.3f}, 匹配: {abs(expected_y - actual_y) < 1e-6}")

    print("数据增强功能测试完成！")

if __name__ == "__main__":
    test_augmentation()