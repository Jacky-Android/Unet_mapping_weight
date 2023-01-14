from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net_se(nn.Module):   #添加了空间注意力和通道注意力
    def __init__(self,img_ch=3,num_classes:int=2):
        super(U_Net_se,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64) #64
        self.Conv2 = conv_block(ch_in=64,ch_out=128)  #64 128
        self.Conv3 = conv_block(ch_in=128,ch_out=256) #128 256
        self.Conv4 = conv_block(ch_in=256,ch_out=512) #256 512
        self.Conv5 = conv_block(ch_in=512,ch_out=1024) #512 1024

        self.se1 = SELayer(channel=64)
        self.se2 = SELayer(channel=128)
        self.se3 = SELayer(channel=256)
        self.se4 = SELayer(channel=512)
        self.se5 = SELayer(channel=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)  #1024 512
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)  

        self.Up4 = up_conv(ch_in=512,ch_out=256)  #512 256
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)  
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)  #256 128
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128) 
        
        self.Up2 = up_conv(ch_in=128,ch_out=64) #128 64
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)  

        self.Conv_1x1 = nn.Conv2d(64,num_classes,kernel_size=1,stride=1,padding=0)  #64


    def forward(self,x):
        # encoding path
        #x1 = self.se1(x)+x
        x1 = self.Conv1(x)
        x1 = self.se1(x1)
        

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2= self.se2(x2)
        
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3= self.se3(x3)
        

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.se4(x4)
        

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.se5(x5)

        # decoding + concat 
        
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return {"out":d1}
