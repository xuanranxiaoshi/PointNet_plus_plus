# -*— coding = utf-8 -*-
# @Time : 2022-04-28 9:30
# @Author : Csc
# @File : pointnet_utils.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    """预测仿射变化矩阵，据此将输入数据对齐到规范空间"""

    def __init__(self, channel):
        super(STN3d, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # 这里的out_channels 的实现是通过多个（与该值一样）的卷积核实现的， 所以等价于 kernel_num
        # 一维卷积层, channel 默认为3
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        # 每一层进行批归一化
        # torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # size: B*D*N
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # B*64*N
        x = F.relu(self.bn2(self.conv2(x)))  # B*128*N
        x = F.relu(self.bn3(self.conv3(x)))  # B*1024*N
        # 在维度2上求每一组的最大值； [0]对应的是值的大小 [1]对应的是值相应的索引
        # keepdim 保持输出的维度与输入一样，否则只会输出一维
        x = torch.max(x, 2, keepdim=True)[0]  # 相当于最大池化, B*1024*1
        # 最后一维度调整为1024
        x = x.view(-1, 1024)  # B*1024

        x = F.relu(self.bn4(self.fc1(x)))  # B*512
        x = F.relu(self.bn5(self.fc2(x)))  # B*256
        x = self.fc3(x)  # B*9

        # todo 计算变换矩阵的原理还不是很清楚
        # 单位矩阵
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)  # B*9
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden  # 加上单位矩阵
        # 转换为3*3的矩阵
        # batch中的每个点云集合都对应一个变换矩阵
        x = x.view(-1, 3, 3)    # B*3*3
        return x


class STNkd(nn.Module):
    """特征变换矩阵，与输入变换矩阵类似"""

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
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """返回通过 poinNet 提取的全局特征或者拼接了局部特征的全局特征"""
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform  # 特征对齐
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        """在输入一维卷积前要交换最后两个维度，因为是在最后一个维度进行卷积操作"""
        B, D, N = x.size()
        trans = self.stn(x)  # 输入对齐 B*3*3
        x = x.transpose(2, 1)  # B*N*D
        if D > 3:
            feature = x[:, :, 3:]  # 特征
            x = x[:, :, :3]  # 坐标
        x = torch.bmm(x, trans)  # 矩阵乘法，相当于对齐的操作 B*N*3
        if D > 3:
            x = torch.cat([x, feature], dim=2)  # 坐标对齐之后拼接上原来的特征

        """MLP 实现为卷积 """
        x = x.transpose(2, 1)  # B*D*N   先交换最后两个维度再进行一维卷积
        x = F.relu(self.bn1(self.conv1(x)))  # 一维卷积操作 B*64*N

        if self.feature_transform:  # 如果要特征对齐
            trans_feat = self.fstn(x)  # B*64*64
            x = x.transpose(2, 1)  # B*N*64
            x = torch.bmm(x, trans_feat)  # 特征对齐 B*N*64
            x = x.transpose(2, 1)  # B*64*N
        else:
            trans_feat = None

        pointfeat = x   # 局部特征 B*64*N
        """MLP"""
        x = F.relu(self.bn2(self.conv2(x)))  # B*128*N
        x = self.bn3(self.conv3(x))  # B*1024*N
        x = torch.max(x, 2, keepdim=True)[0]    # B*1024*1  最大池化
        x = x.view(-1, 1024)    # B*1024
        """上面部分计算的全局特征"""
        if self.global_feat:
            return x, trans, trans_feat     # 全局特征， 输入对齐矩阵， 特征对齐矩阵
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)     # B*1024*1 -> B*1024*N
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # B*1088*N 将全局特征拼接到局部特征中


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]     # 3/64
    I = torch.eye(d)[None, :, :]    # 单元矩阵 1*3*3
    if trans.is_cuda:
        I = I.cuda()
    # torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None) 返回所给tensor的矩阵范数或向量范数
    # 这里默认计算的矩阵范数：每个元素绝对值的平方和
    #  torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)) -> size: B
    loss = torch.mean(  torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))  )
    # 取一个批量的平均值作为损失值-> 这个损失值是为了使特征对齐矩阵接近对角矩阵
    return loss
