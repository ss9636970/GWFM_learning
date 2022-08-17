import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import random
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 用來做第一階段的 graph 和 node attributes predict
class predModel_1(nn.Module):
    def __init__(self, inputD, hD, outputD):
        super(predModel_1, self).__init__()
        self.inputD = inputD
        self.hD = hD
        self.ouputD = outputD

        self.linear = nn.Sequential(nn.Linear(inputD, hD, bias=True),
                                    nn.ReLU(),
                                    nn.Linear(hD, outputD),
                                    nn.Softmax(dim=1))

    def forward(self, x):
        y = self.linear(x)
        return y

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
    def __init__(self, nfeat, dropoutl, layers):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.dropoutl = dropoutl
        self.layers = layers

        self.GraphConvolutionList = nn.ModuleList()
        for i in range(layers - 1):
            s = nn.Sequential(GraphConvolution(nfeat, nfeat),
                              nn.ReLU(),
                              nn.Dropout(dropoutl))
            self.GraphConvolutionList.append(s)
        s = nn.Sequential(GraphConvolution(nfeat, nfeat),
                          nn.ReLU())
        self.GraphConvolutionList.append(s)

    def forward(self, x, adj):
        outputs = x
        for i in range(self.layers):
            outputs = self.GraphConvolutionList[i]([outputs, adj])
        return outputs

# 用來做第二階段，加入 GCN 方法的 model
class predModel_2(nn.Module):
    def __init__(self, inputD, outputD, GCN_layers):
        super(predModel_2, self).__init__()
        self.inputsD = inputD
        self.outputD = outputD
        self.GCN_layers = GCN_layers

        self.GCN = GCN(inputD, 0.1, GCN_layers)
        self.Linear = nn.Linear(5040, outputD)
        self.softmax = nn.Softmax(dim=1)

    def setModel(self, atoms, atomsF):
        self.atoms = atoms
        self.atomsF = atomsF

    def forward(self, trans, lamb):
        attrs = []
        for node in range(trans[0].shape[0]):
            nodes = []
            for atom in range(len(self.atoms)):
                atomAttribute = self.GCN(self.atomsF[atom], self.atoms[atom])
                attr = atomAttribute * trans[atom][node, :].view(-1, 1) * lamb[atom]
                attr = attr.reshape(1, -1)
                nodes.append(attr)
            attr = torch.cat(nodes, dim=1)
            attrs.append(attr)
        attrs = torch.cat(attrs, dim=0)
        outputs = self.Linear(attrs)
        outputs = self.softmax(outputs)
        return outputs