# -*— coding = utf-8 -*-
# @Time : 2022-04-27 15:24
# @Author : Csc
# @File : train_classification.py
# @Software : PyCharm

"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')  # 保存相应的bool值
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')  # 模型名
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')  # 日志文件的名字
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    # 如果当前的子模块含有ReLU的话，
    if classname.find('ReLU') != -1:
        # 设置ReLU层的inplace 属性为True
        # 即进行计算之后原变量的值发生改变
        m.inplace = True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()  # 表明此处是用模型进行测试

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)  # 转置
        pred, _ = classifier(points)    # 前向计算（预测）
        # tensor.data.max(1) 在第1维度寻找最大值； 返回值为两个tensor, 第一个是最大值的列表，第二个是最大值的下标
        pred_choice = pred.data.max(1)[1]   # 预测的结果

        for cat in np.unique(target.cpu()):
            # todo 计算参数不清晰
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # 超参数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    # 以当前的时间作为文件名
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    # exp_dir 此次训练日志文件的根目录

    # 创建checkpoint 文件
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    #  创建日志文件
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    # 初始化日志对象，参数是name, 可以不填
    logger = logging.getLogger("Model")
    # 设置日志的输出的等级
    logger.setLevel(logging.INFO)
    # 输出的格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 指定输出文件
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))  # args.model 是值当前是分类还是分割
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 向日志文件以INFO等级写入指定信息
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    # 数据集日志
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    # 加载数据集
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category  # 分类类型数
    # 动态导入模块
    model = importlib.import_module(args.model)
    # args.model = "pointnet_cls"
    # 将原文件复制到本次训练的日志目录下
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('train_classification.py', str(exp_dir))

    # 获取模型
    classifier = model.get_model(num_class, normal_channel=args.use_normals)  # 模型
    criterion = model.get_loss()  # 损失函数
    # 对classifier 的子模块应用inplace_relu 函数
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        # 将模型放到gpu上
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        # 尝试使用之前训练的模型
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')  # 之前训练的数据
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        # 向日志以INFO等级写入指定信息
        log_string('Use pretrain model')
    except:
        # 从0开始训练
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # 设置优化器
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    # 对学习率进行调整
    # StepLR可以根据超参数gamma每隔固定的step_size就衰减learning_rate一次。
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    # 开始训练
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        """
        如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval();
        model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差;
        对于Dropout, model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接;
        """
        classifier = classifier.train()  # 表明模型现在是在训练， BN层和DROP层正常

        scheduler.step()  # 更新优化器的学习率
        # enumerate(trainDataLoader, 0) 从0开始标号
        """在一轮中对每个批量进行计算"""
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                               smoothing=0.9):
            optimizer.zero_grad()  # 将上一个bath的梯度信息清零

            points = points.data.numpy()  # points 原来为tensor, 这里转化为numpy
            points = provider.random_point_dropout(points)  # 随机drop out 一些点
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])  # 随机缩放点云
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])  # 随机平移点云
            points = torch.Tensor(points)  # 转化为tensor
            # todo 输入数据形状
            # 将第二维和第一维进行转置
            # 原来：b*n*c ； 转置后： b*c*n ?
            points = points.transpose(2, 1)

            if not args.use_cpu:
                # 数据传到gpu
                points, target = points.cuda(), target.cuda()

            # todo 计算的输出
            pred, trans_feat = classifier(points)  # 传入参数前向计算
            loss = criterion(pred, target.long(), trans_feat)  # 计算损失值
            pred_choice = pred.data.max(1)[1]

            # 计算正确率
            #  target.long() 将tensor投射为long类型
            #  tensor.cpu() 将数据传入cpu
            #  tensor.item() 返回张量的值（一个数）
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            global_step += 1  # 训练次数加一

        train_instance_acc = np.mean(mean_correct)  # 这一轮的平均正确率
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():  # 该代码块下 tensor 的require_grad 属性为False, 在反向传播中不会计算梯度
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            # 记录最佳的训练轮数
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),    # model的state_dict变量存放训练过程中需要学习的权重和偏执系数
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 包含state和param_groups的字典对象，而param_groups key对应的value也是一个由学习率，动量等参数组成的一个字典对象。
                }
                # 保存模型等相关参数，利用torch.save()，以及读取保存之后的文件
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
