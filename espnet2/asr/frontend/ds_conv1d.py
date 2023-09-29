#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn


class DSResConv1D(nn.Module):
    def __init__(self, in_channels, layer_num, out_channels, kernel, stride, dilation, **other_params):
        super(DSResConv1D, self).__init__()
        out_channels = expend_params(out_channels, layer_num)
        kernel = expend_params(kernel, layer_num)
        stride = expend_params(stride, layer_num)
        dilation = expend_params(dilation, layer_num)
        in_channel = in_channels
        stack = []
        self.layer_num = layer_num
        for i in range(layer_num):
            stack.append(DSResConvolution1DBlock(in_channels=in_channel, out_channels=out_channels[i], kernel_size=kernel[i],
                                                 stride=stride[i], dilation=dilation[i]))
            in_channel = out_channels[i]
        self.stack = nn.ModuleList(stack)
        self.out_channels = in_channel

    def forward(self, x, length=None):
        for i in range(self.layer_num):
            x, length = self.stack[i](x, length)
        return x, length


class DSResConvolution1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        """
        Args:
            in_channels: Number of channel in input feature
            out_channels: Number of channel in output feature
            kernel_size: Kernel size in D-convolution
            stride: stride in D-convolution
            dilation: dilation factor
            norm_type: BN1d, gLN1d, cLN1d, gLN1d is no causal
        """
        super(DSResConvolution1DBlock, self).__init__()
        # Use `groups` option to implement depth-wise convolution
        # [M, H, K] -> [M, H, K]
        padding = int((kernel_size-1)*dilation/2)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(in_channels)

        if stride >= 1:
            self.d_convolution = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=in_channels,
                                           bias=False, padding_mode='zeros')
            self.length_zoom = lambda x: torch.floor(torch.round((x+2.*padding-dilation*(kernel_size - 1)-1) / stride + 1, decimals=3))
            if stride == 1:
                self.res_downsample = False
            else:
                self.res_downsample = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size//2,
                                                   ceil_mode=False, count_include_pad=True)
        elif stride > 0:
            stride = int(1./stride)
            self.d_convolution = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels,
                                                    kernel_size=kernel_size, stride=stride, padding=padding,
                                                    output_padding=stride-1, groups=in_channels, bias=False,
                                                    dilation=dilation, padding_mode='zeros')
            self.res_downsample = nn.Upsample(size=None, scale_factor=stride, mode='linear', align_corners=False)
            self.length_zoom = lambda x: (x-1)*stride - 2*padding + dilation*(kernel_size - 1) + stride
        else:
            raise ValueError('error stride {}'.format(stride))

        self.s_convolution = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                       padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')

        if in_channels != out_channels:
            self.res_convolution = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                             stride=1, padding=0,  dilation=1, groups=1, bias=False,
                                             padding_mode='zeros')
        else:
            self.res_convolution = None        

    def forward(self, x, length=None):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        res = x
        x = self.relu(x)
        x = self.norm(x)
        x = self.d_convolution(x)
        x = self.s_convolution(x)
        if self.res_convolution:
            res = self.res_convolution(res)
        if self.res_downsample:
            res = self.res_downsample(res)
        
        if length is not None:
            length = self.length_zoom(length)
        return x+res, length


def expend_params(value, length):
    if isinstance(value, list):
        if len(value) == length:
            return value
        else:
            return [value for _ in range(length)]
    else:
        return [value for _ in range(length)]


if __name__ == '__main__':
    checkout_data = torch.ones(3, 1024, 29)
    checkout_length = torch.tensor([15, 25, 29]).long()
    checkout_network = DSResConv1D(in_channels=1024, layer_num=5, out_channels=1536, kernel=5, stride=[1,2,1,2,1], dilation=1)
    print(checkout_network)
    # in_data = [torch.ones(16, 300, 201)]
    checkout_output, checkout_length = checkout_network(checkout_data, checkout_length)
    print(checkout_network.out_channels)
    print(checkout_output.size())
    print(checkout_length)
