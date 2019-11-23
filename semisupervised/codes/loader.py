import sys
import os
import math
import numpy as np
import torch
from torch.autograd import Variable


# 建立不含重复词的词汇表？
class Vocab(object):
    def __init__(self, file_name, cols, with_padding=False):
        # 记录文件里所有词
        self.itos = []
        # 记录某一个词对应的itos的序号
        self.stoi = {}
        self.vocab_size = 0

        if with_padding:
            string = '<pad>'
            self.stoi[string] = self.vocab_size
            self.itos.append(string)
            self.vocab_size += 1

        fi = open(file_name, 'r')
        # 遍历每一行
        for line in fi:
            items = line.strip().split('\t')
            # 遍历每一行的每一列
            for col in cols:
                item = items[col]
                strings = item.strip().split(' ')
                for string in strings:
                    string = string.split(':')[0]
                    # 重复的则不添加
                    if string not in self.stoi:
                        self.stoi[string] = self.vocab_size
                        self.itos.append(string)
                        self.vocab_size += 1
        fi.close()

    def __len__(self):
        return self.vocab_size


class EntityLabel(object):
    # file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1]
    def __init__(self, file_name, entity, label):
        self.vocab_n, self.col_n = entity
        self.vocab_l, self.col_l = label
        # 这里itol是从node的序号映射到label的序号
        # 即node.index to label.index
        self.itol = [-1 for k in range(self.vocab_n.vocab_size)]

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')
            sn, sl = items[self.col_n], items[self.col_l]
            # 同样存的是vocabulary的序号
            n = self.vocab_n.stoi.get(sn, -1)
            l = self.vocab_l.stoi.get(sl, -1)
            if n == -1:
                continue
            self.itol[n] = l
        fi.close()


class EntityFeature(object):
    # file_name = feature_file, entity = [vocab_node, 0], feature = [vocab_feature, 1]
    def __init__(self, file_name, entity, feature):
        self.vocab_n, self.col_n = entity
        self.vocab_f, self.col_f = feature
        # 每个node的序号映射到一个列表，该列表是所有label对应的feature的vocabulary的序号
        self.itof = [[] for k in range(len(self.vocab_n))]
        self.one_hot = []

        fi = open(file_name, 'r')
        for line in fi:
            items = line.strip().split('\t')
            sn, sf = items[self.col_n], items[self.col_f]
            n = self.vocab_n.stoi.get(sn, -1)
            if n == -1:
                continue
            for s in sf.strip().split(' '):
                # 查询某个特征的编号
                f = self.vocab_f.stoi.get(s.split(':')[0], -1)
                w = float(s.split(':')[1])
                if f == -1:
                    continue
                # f表示特征，w表示对应的特征值
                self.itof[n].append((f, w))
        fi.close()

    def to_one_hot(self, binary=False):
        # one_hot矩阵，大小：结点数量 * 特征数量
        self.one_hot = [[0 for j in range(len(self.vocab_f))] for i in range(len(self.vocab_n))]
        # 对每一个结点node
        for k in range(len(self.vocab_n)):
            sm = 0
            # 遍历每一个结点对应的特征
            for fid, wt in self.itof[k]:
                # 这里fid表示feature的编号，wt表示对应特征值
                if binary:
                    wt = 1.0
                # 统计所有特征值之和，由于binary=True（表示所有特征是二值的），因此都为1
                sm += wt
            for fid, wt in self.itof[k]:
                if binary:
                    wt = 1.0
                # 做了个归一化
                self.one_hot[k][fid] = wt / sm


class Graph(object):
    # 初始化获取所有的边信息，存放在edges中
    def __init__(self, file_name, entity, weight=None):
        # entity=[vocab_node, 0, 1]
        self.vocab_n, self.col_u, self.col_v = entity
        self.col_w = weight
        self.edges = []

        self.node_size = -1

        self.eid2iid = None
        self.iid2eid = None

        self.adj_w = None
        self.adj_t = None

        with open(file_name, 'r') as fi:

            for line in fi:
                items = line.strip().split('\t')
                # col_u = 0, col_v = 1
                su, sv = items[self.col_u], items[self.col_v]
                sw = items[self.col_w] if self.col_w != None else None

                # get函数是从字典中查找对应的key（第一个参数），如果找不到返回第二个参数(-1)
                u, v = self.vocab_n.stoi.get(su, -1), self.vocab_n.stoi.get(sv, -1)
                w = float(sw) if sw != None else 1

                # 如果找不到u与v，或者权重为负，则跳过
                if u == -1 or v == -1 or w <= 0:
                    continue

                self.edges += [(u, v, w)]

    def get_node_size(self):
        return self.node_size

    def get_edge_size(self):
        return len(self.edges)

    # 将edges变为对称的，即将所有边去除方向性
    def to_symmetric(self, self_link_weight=1.0):
        vocab = set()
        for u, v, w in self.edges:
            vocab.add(u)
            vocab.add(v)

        pair2wt = dict()
        for u, v, w in self.edges:
            pair2wt[(u, v)] = w

        edges_ = list()
        for (u, v), w in pair2wt.items():
            # 如果相同则跳过，因为最后会统一加上自己指向自己的边
            if u == v:
                continue

            # 对于每一条边，在pair2wt里查询反向边的权重
            w_ = pair2wt.get((v, u), -1)
            # 如果正向的权重较大，则加入两条权重相同、方向不同的边
            # 后续的边就会由于权重较小而被舍弃
            if w > w_:
                edges_ += [(u, v, w), (v, u, w)]
            # 如果相等的话，则加入一条，因为后续还会再加入反向的
            elif w == w_:
                edges_ += [(u, v, w)]

        # 为什么需要将每个结点连向自己？
        for k in vocab:
            edges_ += [(k, k, self_link_weight)]

        d = dict()
        # 统计每个结点相连所有边的权重之和
        for u, v, w in edges_:
            d[u] = d.get(u, 0.0) + w

        # ???这是做啥子
        self.edges = [(u, v, w / math.sqrt(d[u] * d[v])) for u, v, w in edges_]

    def get_sparse_adjacency(self, cuda=True):
        shape = torch.Size([self.vocab_n.vocab_size, self.vocab_n.vocab_size])

        us, vs, ws = [], [], []
        for u, v, w in self.edges:
            us += [u]
            vs += [v]
            ws += [w]
        index = torch.LongTensor([us, vs])
        value = torch.Tensor(ws)
        if cuda:
            index = index.cuda()
            value = value.cuda()
        # 用稀疏矩阵来表示边与权重？
        adj = torch.sparse.FloatTensor(index, value, shape)
        if cuda:
            adj = adj.cuda()

        return adj
