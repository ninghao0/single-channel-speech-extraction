import torch.nn as nn
from nnet.cnns import ChannelwiseLayerNorm, SE_Block
import torch


class Spk_Block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dilation):
        super(Spk_Block, self).__init__()
        self.conv1x1_1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.rule1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.dconv = nn.Conv1d(in_channels, in_channels, 3, groups=in_channels, dilation=dilation, padding=1 * dilation)
        self.conv1x1_2 = nn.Conv1d(mid_channels, in_channels, 1)
        self.se_block = SE_Block(in_channels, in_channels)

    def forward(self, y):
        x = self.dconv(y)
        x = self.norm1(x)
        x = self.conv1x1_1(x)
        x = self.rule1(x)
        x = self.conv1x1_2(x)
        x = self.se_block(x)
        return x + y


class Spk_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, spk_embed_dim):
        super(Spk_Encoder, self).__init__()
        self.norm1 = ChannelwiseLayerNorm(in_channels)
        self.conv1x1_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.spk_block1 = Spk_Block(out_channels, out_channels,  mid_channels, dilation=2 ** 0)
        self.spk_block2 = Spk_Block(out_channels, out_channels,  mid_channels, dilation=2 ** 0)
        self.spk_block3 = Spk_Block(out_channels, out_channels,  mid_channels, dilation=2 ** 0)
        self.spk_block4 = Spk_Block(out_channels, out_channels,  mid_channels, dilation=2 ** 0)
        self.conv1x1_2 = nn.Conv1d(out_channels, spk_embed_dim, kernel_size=1)
        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        x = self.norm1(x)
        x = self.conv1x1_1(x)
        # x = self.maxpool(x)
        x = self.spk_block1(x)
        x = self.maxpool(x)
        x = self.spk_block2(x)
        x = self.maxpool(x)
        x = self.spk_block3(x)
        x = self.maxpool(x)
        x = self.spk_block4(x)
        x = self.maxpool(x)
        x = self.conv1x1_2(x)
        return x
    
class SE_Block2d(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Spk_Block2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dilation):
        super(Spk_Block2d, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), groups=in_channels, dilation=dilation,
                               padding=1 * dilation)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1x1_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1))
        self.rule1 = nn.ReLU()
        self.conv1x1_2 = nn.Conv2d(mid_channels, in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.se_block = SE_Block2d(in_channels, in_channels)

    def forward(self, y):

        x = self.dconv(y)
        x = self.norm1(x)
        x = self.conv1x1_1(x)
        x = self.rule1(x)
        x = self.conv1x1_2(x)
        x = self.se_block(x)

        return x + y


class Spk_Encoder2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, spk_embed_dim_s):
        super(Spk_Encoder2d, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(0, 1))
        self.norm1 = nn.LayerNorm(out_channels)
        self.spk_block1 = Spk_Block2d(out_channels, out_channels, mid_channels, dilation=2 ** 0)
        self.spk_block2 = Spk_Block2d(out_channels, out_channels, mid_channels, dilation=2 ** 0)
        self.spk_block3 = Spk_Block2d(out_channels, out_channels, mid_channels, dilation=2 ** 0)
        self.spk_block4 = Spk_Block2d(out_channels, out_channels, mid_channels, dilation=2 ** 0)
        self.conv1x1_2 = nn.Conv2d(out_channels, spk_embed_dim_s, kernel_size=(3, 3), padding=(0, 1))
        self.maxpool = nn.MaxPool2d(3)

    def forward(self, x):
        x = self.conv1x1_1(x)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 2, 3)
        x = self.spk_block1(x)
        x = self.maxpool(x)
        x = self.spk_block2(x)
        x = self.maxpool(x)
        x = self.spk_block3(x)
        x = self.maxpool(x)
        x = self.spk_block4(x)
        x = self.maxpool(x)
    
        x = self.conv1x1_2(x)
    
        return x


