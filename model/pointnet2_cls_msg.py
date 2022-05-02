# -*— coding = utf-8 -*-
# @Time : 2022-05-02 9:32
# @Author : Csc
# @File : pointnet2_cls_msg.py
# @Software : PyCharm
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # 这里in_channel 是320 = 64 + 128 + 128
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # 这里in_channel 的 640 = 128+256+256
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]    # 法线特征
            xyz = xyz[:, :3, :]     # 坐标信息
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)             # 第一层多尺度采样
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)     # 第二层多尺度采样
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)     # 将所有点作为一个领域，获得全局特征
        x = l3_points.view(B, 1024)                         # 每个点云的全局特征
        """通过权重不同享全连接层"""
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)    # 在最后一维做log_softmax 计算

        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # 将上面（log_softmax）输出中与Label对应的位置的值拿出来去掉负号，求均值
        # label 对应位置的值应该更接近0
        total_loss = F.nll_loss(pred, target)

        return total_loss
