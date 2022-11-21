import torch.nn as nn
import math
from torchmeta.modules import MetaModule, MetaConv2d, MetaBatchNorm2d, MetaLinear, MetaSequential
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        residual = x

        out = self.conv1(x, params=self.get_subdict(params, 'conv1'))
        out = self.bn1(out, params=self.get_subdict(params, 'bn1'))
        out = self.relu(out)

        out = self.conv2(out, params=self.get_subdict(params, 'conv2'))
        out = self.bn2(out, params=self.get_subdict(params, 'bn2'))

        if self.downsample is not None:
            residual = self.downsample(x, params=self.get_subdict(params, 'downsample'))

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(MetaModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        residual = x

        out = self.conv1(x, params=self.get_subdict(params, 'conv1'))
        out = self.bn1(out, params=self.get_subdict(params, 'bn1'))
        out = self.relu(out)

        out = self.conv2(out, params=self.get_subdict(params, 'conv2'))
        out = self.bn2(out, params=self.get_subdict(params, 'bn2'))
        out = self.relu(out)

        out = self.conv3(out, params=self.get_subdict(params, 'conv3'))
        out = self.bn3(out, params=self.get_subdict(params, 'bn3'))

        if self.downsample is not None:
            residual = self.downsample(x, params=self.get_subdict(params, 'downsample'))

        out += residual
        out = self.relu(out)

        return out

