import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
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

class U_Net_eca(nn.Module):   #添加了空间注意力和通道注意力
    def __init__(self,img_ch=3,num_classes:int=2):
        super(U_Net_eca,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64) #64
        self.Conv2 = conv_block(ch_in=64,ch_out=128)  #64 128
        self.Conv3 = conv_block(ch_in=128,ch_out=256) #128 256
        self.Conv4 = conv_block(ch_in=256,ch_out=512) #256 512
        self.Conv5 = conv_block(ch_in=512,ch_out=1024) #512 1024

        self.eca1 = eca_layer(channel=64)
        self.eca2 = eca_layer(channel=128)
        self.eca3 = eca_layer(channel=256)
        self.eca4 = eca_layer(channel=512)
        self.eca5 = eca_layer(channel=1024)

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
        x1 = self.eca1(x1)
        

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2= self.eca2(x2)
        
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3= self.eca3(x3)
        

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.eca4(x4)
        

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.eca5(x5)

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
