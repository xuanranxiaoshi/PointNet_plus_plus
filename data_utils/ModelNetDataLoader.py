'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import argparse

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    """
    点云数据标准化， 均值为0， 方差为1；
    :param pc:
    :return:
    """
    centroid = np.mean(pc, axis=0)  # 求平均值
    pc = pc - centroid  # 减去平均值
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))  # 求标准差
    pc = pc / m  # 标准化
    return pc


def farthest_point_sample(point, npoint):
    """
    对原始数据进行最远点采样
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]  # 每个点坐标
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10 # 相当于无穷大
    farthest = np.random.randint(0, N)  # 随机从N个点中选取一点
    for i in range(npoint):  # 逐一选取npoint个点
        centroids[i] = farthest  # 将最远点加入候选集
        centroid = xyz[farthest, :]  # 最远点的坐标
        dist = np.sum((xyz - centroid) ** 2, -1)    # 计算每个点到该点的距离
        mask = dist < distance
        distance[mask] = dist[mask] # 更新各点到候选集中点的最短距离
        farthest = np.argmax(distance, -1)  # 选出最远的一个点
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    """
    生成 pytorch 的数据库
    """
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category   # 类别的数量

        if self.num_category == 10:     # 根据类别的数量选取不同的文件（文件中存的是每个类别的数量）
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        # 转化为列表
        self.cat = [line.rstrip() for line in open(self.catfile)]
        # 数值化，建立索引
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            # 文件中保存的是训练样本的标号： 如 bathtub_0001
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        # 分理出训练的类型名（文件名）
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # 生成每个样本的文件路径,是一个元组（类型名，一个样本的文件路径）
        self.datapath = [ (shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            # fps 最远点采样
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        # 处理数据
        if self.process_data:
            # 如果不存在该文件夹才会运行
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                # 初始化
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                # 遍历每个样本
                # tqdm 是显示进度条的
                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    # fn为一个元组（类型名， 对应的一个样本文件路径）
                    fn = self.datapath[index]
                    # 类型的标号
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    # 读取文件获得点集；
                    # size: n*b (n各点，每个点有b个属性值)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        # 进行最远点采样至所需节点数
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        # 直接按序选取所需节点数
                        point_set = point_set[0:self.npoints, :]
                    # 存储每个样本的点集和标签
                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls
                # 写入文件
                with open(self.save_path, 'wb') as f:
                    # dump(): 将数据序列化到文件中
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        # 坐标标准化
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.use_normals:
            # 只取坐标部分
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    # parse = argparse.ArgumentParser()
    # parse.add_argument("num_point", type=int)
    # parse.add_argument("use_uniform_sample", type=bool)
    # parse.add_argument("use_normals", type=bool)
    # parse.add_argument("num_category", type=int)
    # args = parse.parse_args()


    # data = ModelNetDataLoader('D:\\code\\data\\modelnet40_normal_resampled', args, split='train', process_data=
    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
