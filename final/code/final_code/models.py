import math
import torch
import torch.nn as nn
from utils import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter

# 
# Class GraphConvolution is from the github
# https://github.com/Megvii-Nanjing/ML-GCN
# 
# Class ResNet is from the github
# https://github.com/OsciiArt/Freesound-Audio-Tagging-2019
# but partially adjusted
# 
# Class GCNResNet and GCN3ResNet are constructed manually using those classes
# 

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.mode = 'train'

        # self.base_model = pretrainedmodels.__dict__['resnet34'](num_classes=num_classes, pretrained=None)
        self.base_model = models.resnet101(pretrained=None)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.last_linear = nn.Linear(self.base_model.layer4[1].conv1.in_channels, num_classes)
        self.last_linear = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )
        self.last_linear2 = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, input):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.gmp(x4).view(bs, -1)
        x = self.last_linear(x)

        return x

    def noisy(self, input):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.gmp(x4).view(bs, -1)
        x = self.last_linear2(x)

        return x


class GCNResNet(nn.Module):
    def __init__(self, num_classes=80, in_channel=300, t=0, adj_file=None):
        super(GCNResNet, self).__init__()

        self.num_classes = num_classes
        self.mode = 'train'

        # self.base_model = pretrainedmodels.__dict__['resnet34'](num_classes=num_classes, pretrained=None)
        self.base_model = models.resnet101(pretrained=None)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.last_linear = nn.Linear(self.base_model.layer4[1].conv1.in_channels, num_classes)
        self.last_linear = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, num_classes)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        self.last_linear2 = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )


    def forward(self, input, inp):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.gmp(x4).view(bs, -1)
        x = self.last_linear(x)

        # inp = inp[0]
        adj = gen_adj(self.A).detach()
        gcn_feature = self.gc1(inp, adj)
        gcn_feature = self.relu(gcn_feature)
        gcn_feature = self.gc2(gcn_feature, adj)

        gcn_feature = gcn_feature.transpose(0, 1)
        x = torch.matmul(x, gcn_feature)

        return x

    def noisy(self, input, inp):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.gmp(x4).view(bs, -1)
        x = self.last_linear2(x)

        # inp = inp[0]
        adj = gen_adj(self.A).detach()
        gcn_feature = self.gc1(inp, adj)
        gcn_feature = self.relu(gcn_feature)
        gcn_feature = self.gc2(gcn_feature, adj)

        gcn_feature = gcn_feature.transpose(0, 1)
        x = torch.matmul(x, gcn_feature)

        return x



class GCN3ResNet(nn.Module):
    def __init__(self, num_classes=80, in_channel=300, adj_file=None):
        super(GCN3ResNet, self).__init__()

        self.num_classes = num_classes
        self.mode = 'train'

        # self.base_model = pretrainedmodels.__dict__['resnet34'](num_classes=num_classes, pretrained=None)
        self.base_model = models.resnet101(pretrained=None)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.last_linear = nn.Linear(self.base_model.layer4[1].conv1.in_channels, num_classes)
        self.last_linear = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, num_classes)
        _adj = gen_A2(num_classes, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        self.last_linear2 = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )


    def forward(self, input, inp):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.gmp(x4).view(bs, -1)
        x = self.last_linear(x)

        # inp = inp[0]
        adj = gen_adj(self.A).detach()
        gcn_feature = self.gc1(inp, adj)
        gcn_feature = self.relu(gcn_feature)
        gcn_feature = self.gc2(gcn_feature, adj)

        gcn_feature = gcn_feature.transpose(0, 1)
        x = torch.matmul(x, gcn_feature)

        return x

    def noisy(self, input, inp):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.gmp(x4).view(bs, -1)
        x = self.last_linear2(x)

        # inp = inp[0]
        adj = gen_adj(self.A).detach()
        gcn_feature = self.gc1(inp, adj)
        gcn_feature = self.relu(gcn_feature)
        gcn_feature = self.gc2(gcn_feature, adj)

        gcn_feature = gcn_feature.transpose(0, 1)
        x = torch.matmul(x, gcn_feature)

        return x

        