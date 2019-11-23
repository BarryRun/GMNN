import sys
import os
import copy
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trainer import Trainer
from gnn import GNNq, GNNp
import loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='/')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--use_gold', type=int, default=1,
                    help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
parser.add_argument('--draw', type=str, default='max',
                    help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)

net_file = opt['dataset'] + '/net.txt'
label_file = opt['dataset'] + '/label.txt'
feature_file = opt['dataset'] + '/feature.txt'
train_file = opt['dataset'] + '/train.txt'
dev_file = opt['dataset'] + '/dev.txt'
test_file = opt['dataset'] + '/test.txt'

# 根据第二个参数，遍历特定列，形成不重复的vocabulary
vocab_node = loader.Vocab(net_file, [0, 1])
vocab_label = loader.Vocab(label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

# 网络中的node数量
opt['num_node'] = len(vocab_node)
# 总共的feature数量
opt['num_feature'] = len(vocab_feature)
# 最终的类别数量
opt['num_class'] = len(vocab_label)
print("num of node:", opt['num_node'])
print("num of feature:", opt['num_feature'])
print("num of class:", opt['num_class'])

# 获取数据中所有的边信息，可以看做生成了一个图
graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
# 获取从node的vocabulary映射到label的vocabulary的数组
label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
# 每个node的序号映射到一个列表，该列表是所有label对应的feature的vocabulary的序号
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
# 将边的数据重新整理，形成无向图（双向的有向图），并且每个点上都有个环
graph.to_symmetric(opt['self_link_weight'])
# 对所有特征做了个归一化
feature.to_one_hot(binary=True)
adj = graph.get_sparse_adjacency(opt['cuda'])

# 读取训练、验证以及测试集（不同的结点集合）
# 其中stoi(string to index)是查询对应的vocab序号
with open(train_file, 'r') as fi:
    idx_train = [vocab_node.stoi[line.strip()] for line in fi]
with open(dev_file, 'r') as fi:
    idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
with open(test_file, 'r') as fi:
    idx_test = [vocab_node.stoi[line.strip()] for line in fi]
idx_all = list(range(opt['num_node']))

inputs = torch.Tensor(feature.one_hot)
target = torch.LongTensor(label.itol)
idx_train = torch.LongTensor(idx_train)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)
inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
target_q = torch.zeros(opt['num_node'], opt['num_class'])
inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
target_p = torch.zeros(opt['num_node'], opt['num_class'])

if opt['cuda']:
    inputs = inputs.cuda()
    target = target.cuda()
    idx_train = idx_train.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    inputs_q = inputs_q.cuda()
    target_q = target_q.cuda()
    inputs_p = inputs_p.cuda()
    target_p = target_p.cuda()

gnnq = GNNq(opt, adj)
trainer_q = Trainer(opt, gnnq)

gnnp = GNNp(opt, adj)
trainer_p = Trainer(opt, gnnp)


def init_q_data():
    # 将inputs拷贝到inputs_q当中
    # 此处inputs为features
    inputs_q.copy_(inputs)
    # 初始化tmp矩阵（训练数据长度 * 类别数目） ==> torch.Size([140, 7])
    temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
    # idx_train是训练数据的node编号列表，target则是node映射到label的数组。因此target[idx_train]查找了某个train数据的label。
    # target[idx_train]就是训练数据的label！！！
    # print(target[idx_train].shape) ==> torch.Size([140])，可以证明上述猜测
    # torch.unsqueeze(target[idx_train], 1) 在target[idx_train]数组的1的位置加了一个维度
    # print(torch.unsqueeze(target[idx_train], 1).shape) ==> torch.Size([140, 1])
    # input.scatter_( dim, index, src)将src中数据根据index中的索引按照dim的方向填进input中
    temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
    # target_q就是每个node所对应的label ==> torch.Size([2708, 7])
    # target_q[idx_train]就是所有训练数据对应的label ==> torch.Size([140, 7])
    # 此处赋值就相当于说明此处2708的结点中只有140个节点的值是已知的，其余都是zeros
    target_q[idx_train] = temp


def update_p_data():
    # 用q的训练结果来做预测
    # 注意区分inputs_q与inputs_p
    # inputs_q是指所有的结点的features
    # inputs_p是所有结点的label，也就是分类类别
    preds = trainer_q.predict(inputs_q, opt['tau'])
    # print(preds.shape)  ==> torch.Size([2708, 7])。是对所有数据均做预测。
    # 然后根据不同的方式来取预测结果
    if opt['draw'] == 'exp':
        inputs_p.copy_(preds)
        target_p.copy_(preds)
    elif opt['draw'] == 'max':
        idx_lb = torch.max(preds, dim=-1)[1]
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    elif opt['draw'] == 'smp':
        # 此时preds是一个概率分布，每一个数据属于不同类别的概率
        idx_lb = torch.multinomial(preds, 1).squeeze(1)
        # print(torch.multinomial(preds, 1).shape) ==> torch.Size([2708, 1])
        # torch.multinomial(preds, 1) 按照矩阵中的权重（可以看做概率）来对每一行随机抽样一次
        # 每一行的类别权重由于已经做了softmax，因此可以看做是概率，这样随机取是合理的
        # print(idx_lb.shape) ==> torch.Size([2708]) squeeze(1)表示压缩为一列
        # 先将inputs_p与target_p用0清空，然后按照idx_lb填充1
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        # 总的来说，就是在每个结点上随机（实际上按权重取）取个类别设为1，其余为0

    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        inputs_p[idx_train] = temp
        target_p[idx_train] = temp


def update_q_data():
    # 注意区分inputs_q与inputs_p
    # inputs_q是指所有的结点的features
    # inputs_p是所有结点的label，也就是分类类别
    # inputs_p会更新，但是inputs_q不会更新
    preds = trainer_p.predict(inputs_p)
    # print(preds.shape) ==> torch.Size([2708, 7])
    # 将p的预测结果的概率分布作为target_q来进行训练
    target_q.copy_(preds)
    # 标注好仍为原有的label
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target[idx_train], 1), 1.0)
        target_q[idx_train] = temp


def pre_train(epoches):
    best = 0.0
    # 先初始化target_q，也其实就是训练数据的label, 其余全部为0
    init_q_data()
    results = []
    for epoch in range(epoches):
        # inputs_q即所有结点的attribute
        # 选择idx_train对应的训练数据，来计算loss值并前馈更新参数
        loss = trainer_q.update_soft(inputs_q, target_q, idx_train)
        # 分别计算在验证集和测试集上的
        _, preds, accuracy_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
        _, preds, accuracy_test = trainer_q.evaluate(inputs_q, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
        if accuracy_dev > best:
            best = accuracy_dev
            state = dict([('model', copy.deepcopy(trainer_q.model.state_dict())),
                          ('optim', copy.deepcopy(trainer_q.optimizer.state_dict()))])
    # 取训练效果最好的模型
    trainer_q.model.load_state_dict(state['model'])
    trainer_q.optimizer.load_state_dict(state['optim'])
    return results


# 对应论文中learning过程，也就是EM的M步
def train_p(epoches):
    # 对所有的结点进行预测，更新input_p与target_p（由于该步是由label预测label，所以两者相同）
    # 也就是求隐变量（未标注变量）的概率分布
    update_p_data()
    results = []
    # 根据更新的target_p与input_p训练GNNp
    for epoch in range(epoches):
        # 计算误差，并前向传播，更新参数
        loss = trainer_p.update_soft(inputs_p, target_p, idx_all)
        # 在验证集与测试集上做evaluate
        _, preds, accuracy_dev = trainer_p.evaluate(inputs_p, target, idx_dev)
        _, preds, accuracy_test = trainer_p.evaluate(inputs_p, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results


# 对应论文中inference过程，也就是EM的E步
def train_q(epoches):
    # 根据GNNp的预测结果更新target_q（此处target_q不同于p，是一个概率分布）
    update_q_data()
    results = []
    # 利用更新过的target_q重新训练GNNq
    for epoch in range(epoches):
        # 计算误差，并前向传播，更新参数
        loss = trainer_q.update_soft(inputs_q, target_q, idx_all)
        # 在验证集与测试集上做evaluate
        _, preds, accuracy_dev = trainer_q.evaluate(inputs_q, target, idx_dev)
        _, preds, accuracy_test = trainer_q.evaluate(inputs_q, target, idx_test)
        results += [(accuracy_dev, accuracy_test)]
    return results


base_results, q_results, p_results = [], [], []
# 先预训练
base_results += pre_train(opt['pre_epoch'])
# 然后EM步交替循环
for k in range(opt['iter']):
    p_results += train_p(opt['epoch'])
    q_results += train_q(opt['epoch'])


def get_accuracy(results):
    best_dev, acc_test = 0.0, 0.0
    for d, t in results:
        if d > best_dev:
            best_dev, acc_test = d, t
    return acc_test


acc_test = get_accuracy(q_results)

print('{:.3f}'.format(acc_test * 100))

if opt['save'] != '/':
    trainer_q.save(opt['save'] + '/gnnq.pt')
    trainer_p.save(opt['save'] + '/gnnp.pt')
