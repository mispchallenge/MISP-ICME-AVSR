import copy
from typing import Optional
from typing import Tuple
from typing import Union
from .vggblock import VGGBlock
import logging
import humanfriendly
import numpy as np
import torch
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from .network_resnet_conv1d import ResNet1D
from torch import nn



def variable_activate(act_type, in_channels=None, **other_params):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'prelu':
        return nn.PReLU(num_parameters=in_channels)
    else:
        raise NotImplementedError('activate type not implemented')

class WavPreEncoder(AbsPreEncoder): #[B,T]->[B,T,512]
    def __init__(
    self,
    conv1d_dim=64,
    conv1d_kernel_size=80,
    conv1d_stride=4,
    res_block_num=2,
    res_stride=[2, 2, 2, 2],
    res_expansion=1,
    res_hidden_channels=[64, 128, 256, 512],
    res_downsample_type="avgpool",
    act_type='prelu',
    ):   
        super().__init__()
        
        default_frontend_setting = {
            "in_channels":1,
            "out_channels":conv1d_dim,
            "kernel_size":conv1d_kernel_size,
            "stride":conv1d_stride,
            "padding":(conv1d_kernel_size-conv1d_stride)//2,#(kernel_size - stride) // 2
            "bias":False,
        }
        self.frontend = nn.Sequential(
            nn.Conv1d(**default_frontend_setting),
            nn.BatchNorm1d(default_frontend_setting["out_channels"]),
            variable_activate(act_type=act_type, in_channels=default_frontend_setting["out_channels"]))

        default_backbone_setting = {
            'block_type': 'basic1d', 'block_num': res_block_num, 'act_type': act_type,
            'hidden_channels': res_hidden_channels, 'stride': res_stride, 'expansion': res_expansion,
            'downsample_type': res_downsample_type} 

        self.backbone = ResNet1D(**default_backbone_setting)
        self.pool = nn.AvgPool1d(10, stride=10) 


    def forward(self,x: torch.Tensor, length: torch.Tensor
    )-> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)# [B,T] -> [B,1,T]
        x = self.frontend(x)# [B,1,T] ->  [B,64,T//4] 
        length = length // 4
        x, length = self.backbone(x, length) # [B,64,T//4] -> [B,512,T//64,] 
        x = self.pool(x).transpose(1,2) # [B,512,T//64]->[B,T//640,512]
        length = length // 10
        if x.size(1) > length.max(): # x may lager than length for 1
            length+=(x.size(1)-length.max())
        return x, length


    def output_size(self) -> int:
        return 512

class featPreEncoder(AbsPreEncoder): #[B,T,D]->[B,T,512]
    def __init__(
    self,
    feat_dim=80,
    conv1d_dim=64,
    conv1d_kernel_size=1,
    conv1d_stride=1,
    res_block_num=2,
    res_stride=[1, 1, 1, 1],
    res_expansion=1,
    res_hidden_channels=[64, 128, 256, 512],
    res_downsample_type="avgpool",
    act_type='prelu',
    ):   
        super().__init__()
   
        default_frontend_setting = {
            "out_channels":conv1d_dim,
            "kernel_size":conv1d_kernel_size,
            "stride":conv1d_stride,
            "bias":False,
            "padding":0,
            "in_channels":feat_dim,
        }
        self.frontend = nn.Sequential(
            nn.Conv1d(**default_frontend_setting),
            nn.BatchNorm1d(default_frontend_setting["out_channels"]),
            variable_activate(act_type=act_type, in_channels=default_frontend_setting["out_channels"]))

        default_backbone_setting = {
            'block_type': 'basic1d', 'block_num': res_block_num, 'act_type': act_type,
            'hidden_channels': res_hidden_channels, 'stride': res_stride, 'expansion': res_expansion,
            'downsample_type': res_downsample_type}

        self.backbone = ResNet1D(**default_backbone_setting)


    def forward(self,x: torch.Tensor, length: torch.Tensor
    )-> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1,2)
        x = self.frontend(x) 
        x, length = self.backbone(x, length)  
        x = x.transpose(1,2)
        return x, length


    def output_size(self) -> int:
        return 512

class VGGfeatPreEncoder(AbsPreEncoder):
    
    def __init__(
        self,
        out_channels = [64,128],
        conv_kernel_size = [3,3],
        pooling_kernel_size = [2,2],
        num_conv_layers = [2,2],
        layer_norm = True,
        input_feat_per_channel=40,
        in_channels=1,
        encoder_output_dim=512):   
        super().__init__()
        self.num_vggblocks = len(out_channels)

        self.encoder_output_dim = encoder_output_dim
        self.conv_layers = nn.ModuleList()
        self.audio_project = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel
        self.pooling_kernel_sizes = pooling_kernel_size
        
        for i in range(self.num_vggblocks): 
            self.conv_layers.append(
                VGGBlock(
                    in_channels,
                    out_channels[i],
                    conv_kernel_size[i],
                    pooling_kernel_size[i],
                    num_conv_layers[i],
                    input_dim=input_feat_per_channel,
                    layer_norm=layer_norm,
                )
            )
            in_channels = out_channels[i]
            input_feat_per_channel = self.conv_layers[-1].output_dim
        self.audio_project.extend(
            [
                nn.Linear(input_feat_per_channel*out_channels[-1], encoder_output_dim,bias=True),
                nn.LayerNorm(encoder_output_dim)
            ]
        )

    def forward(self,src_tokens: torch.Tensor, src_lengths: torch.Tensor
    )-> Tuple[torch.Tensor, torch.Tensor]:
        bsz, max_seq_len, _ = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        x = x.transpose(1, 2).contiguous()
        # (B, C, T, feat)

        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)

        bsz, _, output_seq_len, _ = x.size()

        # (B, C, T, feat) -> (B, T, C, feat) -> (B,T,C*feat)
        x = x.transpose(1, 2)
        x = x.contiguous().view(bsz,output_seq_len,-1)
        
        input_lengths = src_lengths.clone()
        for s in self.pooling_kernel_sizes:
            input_lengths = (input_lengths.float() / s).ceil().long()

        for layer_idx in range(len(self.audio_project)):
            x = self.audio_project[layer_idx](x)
        return x,input_lengths

    def output_size(self) -> int:
        return  self.encoder_output_dim  



if __name__ == "__main__":

    frontend = VGGfeatPreEncoder()
    print(frontend)
    feats = torch.rand(16,55,40) 
    lengths = torch.randint(55,56,(16,))
    output,output_length = frontend(feats,lengths)#[B,T,D]->[B,T,D]
    print(output.shape,output_length)


    frontend = featPreEncoder()
    feats = torch.rand(16,55,80) 
    lengths = torch.randint(50,55,(16,))
    output,output_length = frontend(feats,lengths)#[B,T,D]->[B,T,D]
    print(output.shape,output_length) 

    frontend = WavPreEncoder()
    feats = torch.rand(16,80000) 
    lengths = torch.randint(80000,80001,(16,))
    output,output_length = frontend(feats,lengths)#[B,T]->[B,T,512]
    print(output.shape,output_length) # if input 16k fps -> output 25 fps
