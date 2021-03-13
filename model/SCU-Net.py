###########################################################################
#SCUG-Net: SCU-Net: A deep learning method for segmentation and quantification of breast arterial calcifications on mammography
#Paper-Link: 
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBNPReLU(nn.Module):
    def __init__(self,in_ch, out_ch,  kSize, stride=1):
        super(ConvBNPReLU, self).__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(in_ch, out_ch, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-03)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BNPReLU(nn.Module):
    def __init__(self, out_ch):
        super(BNPReLU, self).__init__()
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-03)
        self.act = nn.PReLU(nOut)
        self.dp = nn.Dropout(p=0.5)
        

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.dp(x)
        return x
    

class ChannelWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch,  kSize, stride=1):
        super(ChannelWiseConv, self).__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(in_ch, out_ch, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class DilatedConv(nn.Module):
    def __init__(self, in_ch, out_ch,  kSize, stride=1, d=1):
        super(DilatedConv, self).__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(in_ch, out_ch,  (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelWiseDilatedConv(nn.Module):
    def __init__(self,in_ch, out_ch,  kSize, stride=1, d=1):
        super(ChannelWiseDilatedConv, self).__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(in_ch, out_ch,  (kSize, kSize), stride=stride, padding=(padding, padding), groups= nIn, bias=False, dilation=d)

    def forward(self, x):
        x = self.conv(x)
        return x

class Fuse(nn.Module):
    """
    the Fuse class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, in_ch, reduction=16):
        super(Fuse, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(in_ch, in_ch // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_ch // reduction, channel),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextBlock(nn.Module):
    def __init__(self,in_ch, out_ch, dilation_rate=2, reduction=16):
        super(ContextBlock, self).__init__()
        self.conv1x1 = ConvBNPReLU(in_ch, out_ch, 3, 2)
        
        self.F_loc = ChannelWiseConv(in_ch, out_ch, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(in_ch, out_ch, 3, 1, dilation_rate)
        
        self.bn = nn.BatchNorm2d(2*out_ch, eps=1e-3)
        self.act = nn.PReLU(2*out_ch)
        self.reduce = Conv(2*out_ch, out_ch,1,1) 
        
        self.F_glo = Fuse(out_ch, reduction)    

    def forward(self, x):
        x = self.conv1x1(x)
        loc = self.F_loc(x)
        sur = self.F_sur(x)

        joi_feat = torch.cat([loc, sur],1) 
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)    
        
        output = self.F_glo(joi_feat) 

        return output


class InputInjection(nn.Module):
    def __init__(self, downsamplingRatio):
        super(InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return input  

class double_conv(nn.Module):
    '''(conv => BN => PReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU())

    def forward(self, x):
        x = self.conv(x)
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


class SCU-Net(nn.Module):
    def __init__(self, classes = 1):
        super(SCU-Net, self).__init__()
        
        self.conv1 = ConvBNPReLU(3, 16, 3, 2)
        self.conv2 = ConvBNPReLU(16, 16, 3, 1)                          
        self.conv3 = ConvBNPReLU(16, 16, 3, 1)
        
        self.cb1 = ContextBlock(16 + 3, 32, dilation_rate=2,reduction=8) 
        self.cb2 = ContextBlock(32+3, 64, dilation_rate=4,reduction=16)
        
        self.b1 = BNPReLU(16 + 3)
        self.b2 = BNPReLU(32 + 3)
        
        self.sample1 = InputInjection(1) 
        self.sample2 = InputInjection(2)  
        
        self.up3 = up(96, 64)
        self.up4 = up(80, 32)
        
        self.outc = outconv(32, classes)
        
    def forward(self, x):
        output0 = self.conv1(x)
        output0 = self.conv2(output0)
        output0 = self.conv3(output0)
        down1 = self.sample1(x)
        down2 = self.sample2(x)
        output0_cat = self.b1(torch.cat([output0, down1], 1))
        output1 = self.cb1(output0_cat)
        output1_cat = self.b2(torch.cat([output1, down2], 1))
        output2 = self.cb2(output1_cat) 
        up1 = self.up3(output2, output1)
        up2 = self.up4(up1, output0)
        up3 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True)
        up3 = self.outc(up3)    
        return up3    
    
