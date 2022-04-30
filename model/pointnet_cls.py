# -*— coding = utf-8 -*-
# @Time : 2022-04-28 9:29
# @Author : Csc
# @File : pointnet_cls.py
# @Software : PyCharm
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer     # 这里加了model路径


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6     # 带有法线特征
        else:
            channel = 3     # 只有坐标
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)     # encoder 提取全局特征 B*1024
        """全连接层进行分类"""
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)     # B*40
        # F.softmax(x,dim=1) 或者 F.softmax(x,dim=0) 在softmax的结果上再做多一次log运算
        # softmax 的范围[0,1] 再取log范围 [-无穷, 0]
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        # 将上面（log_softmax）输出中与Label对应的位置的值拿出来去掉负号，求均值
        # label 对应位置的值应该更接近0
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        # 正则化系数
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
