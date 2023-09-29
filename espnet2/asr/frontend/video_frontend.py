#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn
import numpy as np
import math
import os
user_path = os.path.expanduser('~')
EPS = 1e-16
from .network_resnet_conv2d import ResNet2D, ResNet
from .ds_conv1d import DSResConv1D
from torchvision.transforms import CenterCrop, RandomCrop, RandomHorizontalFlip, Compose


class VideoFrontend(nn.Module):  ##(B, T, 96, 96,3) ->(B,T,512)c
    def __init__(self,random=True,
                channel_input="bgr",
                skip_gray=False,
                size=[88,88],
                downsampling=False,
                hidden_channel_num=64,
                res_layer_block_num=2,
                res_hidden_channels=[ 64, 128, 256, 512],
                res_stride=[ 1, 2, 2, 2 ],
                res_block_type="basic2d",
                res_act_type="prelu",
                res_downsample_type="avgpool",
                use_upsampler=False,
                upsampler_conf={},
                resnettype = "tidy"
                ):
        super(VideoFrontend, self).__init__()
        self.graycropflip=GrayCropFlip(random=random,skip_gray=skip_gray,channel_input=channel_input,size=size)
        self.use_upsampler = use_upsampler
        if self.use_upsampler:
            default_upsampler_conf=dict(in_channels=512, layer_num=2, out_channels=512, kernel=3, stride=[0.5,0.5], dilation=1)
            for key,item in upsampler_conf.items():
                default_upsampler_conf[key] = item
            self.upsampler = DSResConv1D(**default_upsampler_conf)
        self.downsampling = downsampling
        if downsampling:
            self.video_frontend = nn.Sequential(
                nn.Conv3d(1, hidden_channel_num, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(hidden_channel_num), nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)))
        else:
            self.video_frontend = nn.Sequential(
            nn.Conv3d(1, hidden_channel_num, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(hidden_channel_num), nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        backbone_setting = {  
                        "block_type":res_block_type,
                        "block_num": res_layer_block_num,
                        "act_type": res_act_type,
                        "hidden_channels": res_hidden_channels,
                        "stride": res_stride,
                        "expansion": 1,
                        "downsample_type": res_downsample_type,
                        "in_channels": hidden_channel_num}
        self.output_dim = backbone_setting["hidden_channels"][-1]
        if resnettype == "tidy":
            self.resnet = ResNet2D(**backbone_setting)
        elif resnettype == "common":
            self.resnet = ResNet([2, 2, 2, 2],se = True) # only use for exp/vsr_pretrainmodel/nopain_lipresnet.pth
    def output_size(self) -> int:
        return self.output_dim

    def forward(self, x,x_len):    
        x,_=self.graycropflip(x) #”bgr“ #(B, T, 96, 96,3) ->  (B, T, 88, 88) 
        assert x.dim() == 4, f'shape error: input must  (B, T, 88, 88)'
        B, T, _, _ = x.size() #(B, T, 88, 88)  
        if self.downsampling:
            T = ((T-1)//2-1)//2
            x_len = ((x_len-1)//2-1)//2
        x = x.unsqueeze(1) #(B, 1, T, 88, 88) 
        x = self.video_frontend(x)  #(B, 64, T, 88, 88) #only time connnect are applied here
        x = x.transpose(1, 2).contiguous()  #(B, T, 64, 88, 88) 
        x = x.view(-1, 64, x.size(3), x.size(4)) #(B*T, 64, 88, 88) 
        x,x_len = self.resnet(x,x_len) #(B*T,64,88,88)->(B*T,64,22,22)->(B*T,128,11,11)->(B*T,256,6,6)->(B*T,512,3,3)->(B*T,512)
        x = x.view(B, T, -1) #(B,T,512) 
        if self.use_upsampler:
            x,x_len = self.upsampler(x.transpose(1,2),x_len)
            x = x.transpose(1,2)
        return x,x_len

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class GrayCropFlip(nn.Module): #B,T,96,96,3-> #B,T,size,size
    def __init__(self, channel_input='bgr', size=None, random=False, skip_gray=False, **other_params):
        #size=88,88
        super(GrayCropFlip, self).__init__()
        self.skip_gray = skip_gray
        if not self.skip_gray:
            self.channel2idx = {channel_input[i]: i for i in range(len(channel_input))}
        if size is not None:
            self.random = random
            #padding means to pad value around the image if it is needed
            self.train_transform = Compose([
                RandomCrop(size=size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
                RandomHorizontalFlip(p=0.5)])
            self.eval_transform = Compose([CenterCrop(size=size)])

    def forward(self, x, length=None):
        #1.conver bgr to gray(sum each channel with weighted) #B,T,96,96,3-> #B,T,96,96
        if not self.skip_gray:
            assert x.shape[-1] == 3, 'shape error: input must have r,g,b 3 channels, but got {}'.format(x.shape)
            x_split = x.split(1, dim=-1)
            gray_frames = 0.114 * x_split[self.channel2idx['b']] + 0.587 * x_split[
                self.channel2idx['g']] + 0.299 * x_split[self.channel2idx['r']]
            x = gray_frames.sum(dim=-1)
        # apply random cut and randhorizontalFlip when train (#B,T,96,96->B,T,size,size);when eval apply center cut
        if hasattr(self, 'random'):
            x = self.train_transform(x) if self.training and self.random else self.eval_transform(x)
        return x, length

        pass

if __name__ == "__main__":
    frontend = VideoFrontend(downsampling=False,use_upsampler=True)
    feats = torch.rand(16,55, 96, 96, 3)  #121, 96, 96, 3
    lengths = torch.randint(55,56,(16,))
    output,output_length = frontend(feats,lengths)#[B,T,D]->[B,T,D]
    print(output.shape,output_length) 