import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    # adj表示图结构
    def __init__(self, opt, adj):
        super(GraphConvolution, self).__init__()
        self.opt = opt
        self.in_size = opt['in']
        self.out_size = opt['out']
        self.adj = adj
        # 这里的weight是不是可以理解为representation
        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.reset_parameters()

    # 权重初始化
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        m = torch.mm(x, self.weight)
        m = torch.spmm(self.adj, m)
        return m
