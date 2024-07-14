#!/usr/bin/env python
import torch
import torch as th
import torch.nn as nn

from nnet.norm import ChannelwiseLayerNorm, GlobalLayerNorm

class Conv1D(nn.Conv1d):
    """
    1D Conv based on nn.Conv1d for 2D or 3D tensor
    Input: 2D or 3D tensor with [N, L_in] or [N, C_in, L_in]
    Output: Default 3D tensor with [N, C_out, L_out]
            If C_out=1 and squeeze is true, return 2D tensor
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} require a 2/3D tensor input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D Transposed Conv based on nn.ConvTranspose1d for 2D or 3D tensor
    Input: 2D or 3D tensor with [N, L_in] or [N, C_in, L_in]
    Output: 2D tensor with [N, L_out]
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} require a 2/3D tensor input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))

        # squeeze the channel dimension 1 after reconstructing the signal
        return th.squeeze(x, 1)

class TCNBlock(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
    Input: 3D tensor with [N, C_in, L_in]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(TCNBlock, self).__init__()
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(conv_channels, elementwise_affine=True) if not causal else ( 
            ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(conv_channels, elementwise_affine=True) if not causal else ( 
            ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y

class TCNBlock_Spk(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
        The first tcn block takes additional speaker embedding as inputs
    Input: 3D tensor with [N, C_in, L_in]
    Input Speaker Embedding: 2D tensor with [N, D]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(self,
                 in_channels=256,
                 spk_embed_dim=100,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(TCNBlock_Spk, self).__init__()
        self.conv1x1 = Conv1D(in_channels+spk_embed_dim, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(conv_channels, elementwise_affine=True) if not causal else ( 
            ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(conv_channels, elementwise_affine=True) if not causal else ( 
            ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad
        self.dilation = dilation

    def forward(self, x, aux):
        # Repeatedly concated speaker embedding aux to each frame of the representation x
        T = x.shape[-1]
        aux = th.unsqueeze(aux, -1)
        aux = aux.repeat(1,1,T)
        y = th.cat([x, aux], 1)
        # y = self.channel_shuffle(y, 2)
        y = self.conv1x1(y)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y

    # def channel_shuffle(x, groups):
    #     batchsize, num_channels, length = x.data.size()
    #     channels_per_group = num_channels // groups
    #     # num_channels = groups * channels_per_group
    #
    #     # grouping, 通道分组
    #     # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    #     x = x.view(batchsize, groups, channels_per_group, length)
    #
    #     # channel shuffle, 通道洗牌
    #     x = th.transpose(x, 1, 2).contiguous()
    #     # x.shape=(batchsize, channels_per_group, groups, height, width)
    #     # flatten
    #     x = x.view(batchsize, -1, length)
    #
    #     return x
    
    
class TCNBlock_next(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
    Input: 3D tensor with [N, C_in, L_in]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(TCNBlock_next, self).__init__()
        self.causal = causal
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
                dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.norm1 = GlobalLayerNorm(in_channels, elementwise_affine=True) if not causal else (
            ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        # self.prelu2 = nn.PReLU()
        # self.norm2 = GlobalLayerNorm(conv_channels, elementwise_affine=True) if not causal else (
        #     ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.dconv_pad = dconv_pad

    def forward(self, x):

        y = self.dconv(x)
        y = self.norm1(y)
        y = self.conv1x1(y)
        y = self.prelu1(y)
        y = self.sconv(y)
        y += x
        return y


class TCNBlock_Spk_next(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
        The first tcn block takes additional speaker embedding as inputs
    Input: 3D tensor with [N, C_in, L_in]
    Input Speaker Embedding: 2D tensor with [N, D]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(self,
                 in_channels=256,
                 spk_embed_dim=100,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(TCNBlock_Spk_next, self).__init__()
        self.causal = causal
        self.dilation = dilation
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
                dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(
            in_channels + spk_embed_dim,
            in_channels + spk_embed_dim,
            kernel_size,
            groups=in_channels + spk_embed_dim,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.norm1 = GlobalLayerNorm(in_channels + spk_embed_dim, elementwise_affine=True) if not causal else (
            ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        self.conv1x1 = Conv1D(in_channels + spk_embed_dim, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        # self.prelu2 = nn.PReLU()
        # self.norm2 = GlobalLayerNorm(conv_channels, elementwise_affine=True) if not causal else (
        #     ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.dconv_pad = dconv_pad


    def forward(self, x, aux):
        # Repeatedly concated speaker embedding aux to each frame of the representation x
        T = x.shape[-1]
        aux = th.unsqueeze(aux, -1)
        aux = aux.repeat(1, 1, T)
        y = th.cat([x, aux], 1)
        y = self.dconv(y)
        y = self.norm1(y)
        y = self.conv1x1(y)
        y = self.prelu1(y)
        y = self.sconv(y)
        y += x
        return y


class ResBlock(nn.Module):
    """
    Resnet block for speaker encoder to obtain speaker embedding
    ref to 
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://github.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    """
    def __init__(self, in_dims, out_dims):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        # self.seblock = SE_Block(out_dims)
        self.maxpool = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        # y = self.seblock(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return self.maxpool(y)
    
    
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=1):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        GlobalLayerNorm(in_channel),
        nn.PReLU(),
        nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers, causal=False):
        super(dense_block, self).__init__()
        self.causal = causal
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x



if __name__ == '__main__':
    x = torch.randn(1,10,3)
    print(x)

