import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNectBlock(nn.Module):
    r"""ConvNeXt ConvNectBlock. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class UpConvNext(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.upscale_factor = 2
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm = LayerNorm(ch_out, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.act(x)
        return x


class UpConvNext2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.upscale_factor = 2
        self.pixel = nn.PixelShuffle(upscale_factor=self.upscale_factor)
        self.up = nn.ConvTranspose2d(
            ch_in // self.upscale_factor ** 2,
            ch_out,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm = LayerNorm(ch_out, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pixel(x)
        x = self.up(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.act(x)
        return x


class ConvNext_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv_next = nn.Sequential(ConvNectBlock(dim=ch_out))
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        return self.conv_next(x)


class Unet_ConvNext(nn.Module):
    """
    Version 1 using jast ConvNext_block. No bach normalization and no relu.
    """

    def __init__(self, img_ch=3, num_classes:int=2, channels=64):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.0625)

        self.Conv1 = ConvNext_block(ch_in=img_ch, ch_out=channels)
        
        self.Conv2 = ConvNext_block(ch_in=channels, ch_out=channels * 2)
        self.Conv2x = ConvNext_block(ch_in=channels*2, ch_out=channels * 2)
        self.Conv3 = ConvNext_block(ch_in=channels * 2, ch_out=channels * 4)
        self.Conv3x = ConvNext_block(ch_in=channels * 4, ch_out=channels * 4)
        self.Conv4 = ConvNext_block(ch_in=channels * 4, ch_out=channels * 8)
        self.Conv4x = ConvNext_block(ch_in=channels * 8, ch_out=channels * 8)
        self.Conv5 = ConvNext_block(ch_in=channels * 8, ch_out=channels * 16)
        self.Conv5x = ConvNext_block(ch_in=channels * 16, ch_out=channels * 16)

        self.Up5 = UpConvNext2(ch_in=channels * 16, ch_out=channels * 8)
        self.Up_conv5 = ConvNext_block(ch_in=channels * 16, ch_out=channels * 8)
        self.Up_conv5x = ConvNext_block(ch_in=channels * 8, ch_out=channels * 8)

        self.Up4 = UpConvNext2(ch_in=channels * 8, ch_out=channels * 4)
        self.Up_conv4 = ConvNext_block(ch_in=channels * 8, ch_out=channels * 4)
        self.Up_conv4x = ConvNext_block(ch_in=channels * 4, ch_out=channels * 4)

        self.Up3 = UpConvNext2(ch_in=channels * 4, ch_out=channels * 2)
        self.Up_conv3 = ConvNext_block(ch_in=channels * 4, ch_out=channels * 2)
        self.Up_conv3x = ConvNext_block(ch_in=channels * 2, ch_out=channels * 2)

        self.Up2 = UpConvNext2(ch_in=channels * 2, ch_out=channels)
        self.Up_conv2 = ConvNext_block(ch_in=channels * 2, ch_out=channels)
        self.Up_conv2x = ConvNext_block(ch_in=channels, ch_out=channels)


        self.Conv_1x1 = nn.Conv2d(
            channels, num_classes, kernel_size=1, stride=1, padding=0
        )
        self.last_activation = nn.Hardtanh()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        
        

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.Conv2x(x2)
        x2 = self.dropout(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.Conv3x(x3)
        x3 = self.dropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.Conv4x(x4)
        x4 = self.dropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.Conv5x(x5)
        x5 = self.dropout(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = self.Up_conv5x(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.Up_conv4x(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.Up_conv3x(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.Up_conv2x(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.last_activation(d1)

        return {"out":d1}
