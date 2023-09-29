#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import math
import torch
import torch.nn as nn
from .network_common_module import variable_activate, DownSample2d, expend_params


class ResNet2D(nn.Module):
    def __init__(
            self, block_type='basic', block_num=2, in_channels=64, hidden_channels=256, stride=1, act_type='relu',
            expansion=1, downsample_type='norm', **other_params):
        super(ResNet2D, self).__init__()
        self.layer_num = 4
        type2block = {'basic2d': BasicBlock2D, 'bottleneck2d': BottleneckBlock2D}
        hidden_channels_of_layers = expend_params(value=hidden_channels, length=self.layer_num)
        stride_of_layers = expend_params(value=stride, length=self.layer_num)
        act_type_of_layers = expend_params(value=act_type, length=self.layer_num)
        expansion_of_layers = expend_params(value=expansion, length=self.layer_num)
        downsample_type_of_layers = expend_params(value=downsample_type, length=self.layer_num)

        in_planes = in_channels
        for layer_idx in range(self.layer_num):
            blocks = []
            for block_idx in range(expend_params(value=block_num, length=self.layer_num)[layer_idx]):
                blocks.append(
                    type2block[block_type](
                        in_channels=in_planes, hidden_channels=hidden_channels_of_layers[layer_idx],
                        stride=stride_of_layers[layer_idx] if block_idx == 0 else 1,
                        act_type=act_type_of_layers[layer_idx], expansion=expansion_of_layers[layer_idx],
                        downsample_type=downsample_type_of_layers[layer_idx]))
                in_planes = int(hidden_channels_of_layers[layer_idx] * expansion_of_layers[layer_idx])
            setattr(self, 'layer{}'.format(layer_idx), nn.Sequential(*blocks))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                pass
        # if self.gamma_zero:
        #     for m in self.modules():
        #         if isinstance(m, BasicBlock):
        #             m.norm2.weight.data.zero_()

    def forward(self, x, length=None):
        for layer_idx in range(self.layer_num):
            x = getattr(self, 'layer{}'.format(layer_idx))(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x, length


class BasicBlock2D(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, stride=1, act_type='relu', expansion=1, downsample_type='norm',
            **other_params):
        super(BasicBlock2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=stride, padding=1,
                bias=False),
            nn.BatchNorm2d(hidden_channels),
            variable_activate(act_type=act_type, in_channels=hidden_channels))

        out_channels = hidden_channels * expansion
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.act2 = variable_activate(act_type=act_type, in_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = DownSample2d(
                in_channels=in_channels, out_channels=out_channels, stride=stride, downsample_type=downsample_type)
        else:
            pass

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act2(out + residual)
        return out


class BottleneckBlock2D(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, stride=1, act_type='relu', expansion=1, downsample_type='norm',
            **other_params):
        super(BottleneckBlock2D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            variable_activate(act_type=act_type, in_channels=hidden_channels))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            variable_activate(act_type=act_type, in_channels=hidden_channels))

        out_channels = int(hidden_channels * expansion)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.act3 = variable_activate(act_type=act_type, in_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = DownSample2d(
                in_channels=in_channels, out_channels=out_channels, stride=stride, downsample_type=downsample_type)
        else:
            pass

    def forward(self, x):
        residual = self.downsample(x) if hasattr(self, 'downsample') else x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act3(out + residual)
        return out


## only use for exp/vsr_pretrainmodel/nopain_lipresnet.pth
class ResNet(nn.Module):
    # layers: [2, 2, 2, 2]
    # se: True 
    def __init__(self, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x, length=None):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x, length        


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        
        if(self.se):
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes//16)
            self.conv4 = conv1x1(planes//16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        if(self.se):
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()
            
            out = out * w
        
        out = out + residual
        out = self.relu(out)

        return out
