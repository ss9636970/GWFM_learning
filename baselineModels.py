import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import random
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        attributes = inputs[0]
        adj = inputs[1]
        n = adj.shape[0]
        A = adj + torch.eye(n).to(device)
        D = torch.diag(torch.sum(adj, dim=0))
        L = D - A
        support = torch.mm(attributes, self.weight) / self.out_features
        output = torch.mm(L, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, dropoutl):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.dropoutl = dropoutl

        self.g1 = nn.Sequential(GraphConvolution(nfeat, nfeat + 12),
                            nn.ReLU(),
                            nn.Dropout(dropoutl))

        self.g2 = nn.Sequential(GraphConvolution(nfeat + 12, nfeat + 23),
                            nn.ReLU(),
                            nn.Dropout(dropoutl))

        self.g3 = nn.Sequential(GraphConvolution(nfeat + 23, 38),
                            nn.Softmax(dim=1))



    def forward(self, x, adj):
        outputs = x
        outputs = self.g1([outputs, adj])
        outputs = self.g2([outputs, adj])
        outputs = self.g3([outputs, adj])
        return outputs