# -*— coding = utf-8 -*-
# @Time : 2022-04-30 17:16
# @Author : Csc
# @File : pointnet2_utils.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


# 打印时间
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()  # 返回当前时间戳


def pc_normalize(pc):
    """点云数据进行归一化"""
    l = pc.shape[0]  # 假设N*D
    centroid = np.mean(pc, axis=0)  # 求个坐标的均值得到的中间点
    pc = pc - centroid  # 以中心点为参考
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 每个点到中心点的距离， 找最大的距离，最大标准差
    pc = pc / m  # 归一化，z-score 标准化方法， 即(x-mean)/std
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    计算src 和 dst 两个点集中任意两点之间的距离的平方
    src * dst^T = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B*N*C * B*C*M = B*N*M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # B*N*1
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # B*1*M
    return dist


def index_points(points, idx):
    """
    根据索引 idx 采样点云中指定的点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device  # 这里point应该是一个tensor
    B = points.shape[0]
    view_shape = list(idx.shape)  # [B， S]
    view_shape[1:] = [1] * (len(view_shape) - 1)  # [B, 1]
    repeat_shape = list(idx.shape)  # [B, S]
    repeat_shape[0] = 1  # [1, S]
    # PyTorch中的repeat()函数可以对张量进行重复扩充。
    # 当参数只有两个时：（列的重复倍数，行的重复倍数）。1表示不重复
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # [B, S]
    # 从points当中取出每个batch_indices对应索引的数据点
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样
    Input:
        xyz: pointcloud data, [B, N, 3]  仅坐标
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint] 采样点的索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 初始化占位 B*npoint
    distance = torch.ones(B, N).to(device) * 1e10  # 距离矩阵，用来记录所有点到已经选取的点的最小距离， 初始化为大值 B*N
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 首先随机在每个点云中选取一个点作为最远点 B*1
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # 0-（B-1）的数组
    for i in range(npoint):  # 依次采样
        centroids[:, i] = farthest  # 赋值最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 取出最远点的坐标 B*1*3
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 点云中其他点到选取的最远点之间的距离 B*N
        mask = dist < distance
        distance[mask] = dist[mask]  # 更新distance
        farthest = torch.max(distance, -1)[1]  # 选取距离最远的点更新 farthest
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    寻找各球形领域中的采样点
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]    S个球形领域的中心点坐标
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # B*S*N
    sqrdists = square_distance(new_xyz, xyz)  # B*S*N 任意两点之间的距离的平方
    group_idx[sqrdists > radius ** 2] = N  # 将距离大于半径的地方赋值为N(其他值为0到N-1， 所以N相当于最大值)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # 升序排序，并截取前nsample 个索引
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])  # 全以第一个点代替
    mask = group_idx == N
    group_idx[mask] = group_first[mask]  # 处理距离小于半径的点少于nsample的情况，用第一个点代替
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    主要用于将整个点云分散成局部的 group
    Input:
        npoint: 中点的个数（球领域数量）
        radius: 邻域半径
        nsample: 邻域中的采样点数
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # 最远点采样获得索引 B*npoint
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C] 通过索引采样获得最远点
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample] 返回在每个邻域内采样的点的索引
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C] # 通过索引获得采样点
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # 采样点减去中心点的值

    if points is not None:  # 如果点云除了坐标信息还有其他特征，则将特征拼接到坐标之后
        grouped_points = index_points(points, idx)  # 从原始点云中挑选出的采样得到的点
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D] 在最后一维度进行拼接
    else:
        new_points = grouped_xyz_norm  # 如果没有其他特征, 那么直接放回采样点的坐标
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx  # 还有邻域采样点的索引， 最远点的索引
    else:
        return new_xyz, new_points  # 中心点； 邻域采样后的点


def sample_and_group_all(xyz, points):
    """
    将所有点作为一个 group
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)  # B*1*C 初始化中点为（0,0,0）
    grouped_xyz = xyz.view(B, 1, N, C)  # 每个点云中只有一个邻域
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """ 对输入点云进行采样，然后提取领域的全局特征 """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        # 例如：npoint=128,radius=0.4,nsample=64,in_channle=128+3,mlp=[128,128,256],group_all=False
        # 128=npoint:points sampled in farthest point sampling
        # 0.4=radius:search radius in local region
        # 64=nsample:how many points inn each local region
        # [128,128,256]=output size for MLP on each point

        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint  # 中心点、领域数量
        self.radius = radius  # 每个邻域的采样半径
        self.nsample = nsample  # 每个邻域中点的数量
        self.mlp_convs = nn.ModuleList()  # 卷积层
        self.mlp_bns = nn.ModuleList()  # 批归一化层
        last_channel = in_channel  # 上一层输入通道大小
        for out_channel in mlp:
            """这里卷积核的大小是1"""
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # B*N*C
        if points is not None:  # 还有其他特征
            points = points.permute(0, 2, 1)  # B*N*D
        """采样"""
        if self.group_all:  # 是否将点云中的所有点作为一个领域
            new_xyz, new_points = sample_and_group_all(xyz, points)  # (B, 1, C) (B, 1, N, D+C)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        """提取特征"""
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        # todo 这里卷积操作不是很理解
        # Conv2d input输入的四维张量[N, C, H, W]， C表示channel
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]  # 获得对应的批归一化层
            new_points = F.relu(bn(conv(new_points)))  # 前向计算
        # 卷积之后new_points的大小为：B*out_channel*nsample*npoint
        new_points = torch.max(new_points, 2)[0]  # 第三维求最大值， B*out_chane*npoint
        new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, npoint]
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """MSG方法的局部特征提取"""

    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        """
        :param npoint: 中心点数
        :param radius_list: 这里的邻域半径是一个列表
        :param nsample_list: 同样采样点数也是一个与半径对应的列表
        :param in_channel: 输入特征维度
        :param mlp_list:  每个感知机每一层的维度
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3  # 特征维度+坐标维度
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # B*N*C
        if points is not None:
            points = points.permute(0, 2, 1)  # B*N*D

        B, N, C = xyz.shape
        S = self.npoint  # 领域数
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  # 最远点采样获得邻域中心点
        new_points_list = []  # 存放结果
        for i, radius in enumerate(self.radius_list):  # 每一个邻域半径进行采样
            K = self.nsample_list[i]  # 当前采样点数
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  # 根据中心点进行采样，返回采样索引 B*S*nsample
            grouped_xyz = index_points(xyz, group_idx)  # 获取采样点， B*S*nsample*C
            grouped_xyz -= new_xyz.view(B, S, 1, C)  # 每个领域都减去中心点
            if points is not None:
                grouped_points = index_points(points, group_idx)  # 有特征的话，提取获得有特征的采样点
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # 将特征和坐标拼接在一起
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            """获得局部区域的全局特征"""
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # 拼接不同半径下的点云特征
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """
    特征传递，通过线性插值和的MLP实现，主要用于分割
    """

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)  # B*N*C
        xyz2 = xyz2.permute(0, 2, 1)  # B*S*C

        points2 = points2.permute(0, 2, 1)  # B*S*D
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        """如果下一层点的数量是1,那么将该点重复N次"""
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)  # 获得两个点集中任意两个点的之间的距离， B*N*S
            dists, idx = dists.sort(dim=-1)  # 最后一个维度按距离排序
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]  为N中的每一个点在S中选取最近的三个点

            dist_recip = 1.0 / (dists + 1e-8)  # 取倒数，距离越远权重越小 B*N*3
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # 在第三维求和 B*N*1
            weight = dist_recip / norm  # 归一化求权重 B*N*3
            # 从第二个点集中提取出每个点最近的三个点：index_points(points2, idx) B*N*3*D
            # 对应点乘以对应权重，并求和: B*N*D
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # B*N*D
            # 拼接上下采样的前对应点的特征
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # B*D*N

        """用卷积模拟全连接， 重新计算点的特征"""
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
