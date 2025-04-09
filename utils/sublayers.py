import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module): 
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out

    
class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, ncla, dropout, use_bn):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid*2, dropout, bias=True)
        self.Linear2 = Linear(nhid*2, nhid, dropout, bias=True)
        self.Linear3 = Linear(nhid, ncla, dropout, bias=True)
        # self.Linear = Linear(nfeat, ncla, dropout, bias=True)

        self.use_bn = use_bn 
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid*2)
            self.bn3 = nn.BatchNorm1d(nhid)

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(self.Linear1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(self.Linear2(x))
        if self.use_bn:
            x = self.bn3(x)
        x = F.relu(self.Linear3(x))
        # x = self.Linear(x)

        return x
