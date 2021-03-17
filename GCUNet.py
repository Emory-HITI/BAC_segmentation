
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# __all__ = ["CGUNet"]

class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
        self.dp = nn.Dropout(p=0.5)
        

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        output = self.dp(output)
        return output
    

class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output
    
class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups= nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)  #  size/2, channel: nIn--->nOut
        
        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
        
        self.bn = nn.BatchNorm2d(2*nOut, eps=1e-3)
        self.act = nn.PReLU(2*nOut)
        self.reduce = Conv(2*nOut, nOut,1,1)  #reduce dimension: 2*nOut--->nOut
        
        self.F_glo = FGlo(nOut, reduction)    

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur],1)  #  the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)     #channel= nOut
        
        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        return output


class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n= int(nOut/2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1) # local feature
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate) # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo= FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = torch.cat([loc, sur], 1) 

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output  = input + output
        return output

class InputInjection(nn.Module):
    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input

class OutputInjection(nn.Module):
    def __init__(self, upsamplingRatio):
        super().__init__()
        self.up = nn.ModuleList()
        for i in range(0, upsamplingRatio):
            self.up.append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners = False))
    def forward(self, input):
        for up in self.up:
            input = up(input)
        return input
    

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x



class CGUNet(nn.Module):
    def __init__(self, classes = 1):
        super(CGUNet, self).__init__()
        self.level1_0 = ConvBNPReLU(3, 16, 3, 2)
        self.level1_1 = ConvBNPReLU(16, 16, 3, 1)                          
        self.level1_2 = ConvBNPReLU(16, 16, 3, 1)
        self.level2_0 = ContextGuidedBlock_Down(16 + 3, 32, dilation_rate=2,reduction=8) 
        self.b1 = BNPReLU(16 + 3)
        self.b2 = BNPReLU(32 + 3)
        
        self.sample1 = InputInjection(1)  #down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  #down-sample for Input Injiection, factor=4
        
        self.level3_0 = ContextGuidedBlock_Down(32+3, 64, dilation_rate=4,reduction=16)
        self.level3_1 = ContextGuidedBlock_Down(32 , 64, dilation_rate=2,reduction=8) 

        self.up3 = up(96, 64)
        self.up4 = up(80, 32)
        self.outc = outconv(32, classes)
        self.m = nn.Sigmoid()
        
    def forward(self, x):
        output0 = self.level1_0(x)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        down1 = self.sample1(x)
        down2 = self.sample2(x)
        output0_cat = self.b1(torch.cat([output0, down1], 1))
        output1 = self.level2_0(output0_cat)
        output1_cat = self.b2(torch.cat([output1, down2], 1))
        output2 = self.level3_0(output1_cat)
        up1 = self.up3(output2, output1)
        up2 = self.up4(up1, output0)
        up3 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True)
        up3 = self.outc(up3)
       
        return up3    
    
