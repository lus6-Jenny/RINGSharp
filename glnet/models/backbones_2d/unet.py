import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def last_conv_block(in_channels, mid_channels, bn=False):
    if bn:
        block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
        )
    else:
        block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, stride=1),
                nn.Sigmoid()
        )
    return block


class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)


# ------ TransNet Architecture ------
class ECALayer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
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


class TransNet(nn.Module):
    def __init__(self, in_channels, bn=False):
        super(TransNet, self).__init__()
        self.conv1 = conv_block_unet(in_channels, 64, bn=bn)
        self.eca1 = ECALayer(64)
        self.conv2 = conv_block_unet(64, 64, bn=bn)
        self.eca2 = ECALayer(64)
        self.conv_last = conv2d(64, 1, 1, 1, 0, bn=bn)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.eca1(x)
        x = self.conv2(x)
        x = self.eca2(x)
        x = self.conv_last(x)
        
        return x
   

# ------ UNet Architecture ------
class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bn=False, is_circular=False):
        super(conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.is_circular = is_circular
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

    def forward(self, x):
        if self.is_circular:
            x = F.pad(x, pad=[0, self.kernel_size-1, 0, self.kernel_size-1] , mode='circular')
        out = self.conv(x)
        return out


class conv_block_unet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=False, is_circular=False):
        super(conv_block_unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_circular = is_circular
        if is_circular:
            padding = 0
        else:
            padding = 1
        if bn:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.ReLU(inplace=True))     

    def forward(self, x):
        if self.is_circular:
            x = F.pad(x, pad=[0, self.kernel_size-1, 0, self.kernel_size-1] , mode='circular')
        out = self.conv(x)
        return out


class conv_last_block_unet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=False, is_circular=False):
        super(conv_last_block_unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_circular = is_circular
        if is_circular:
            padding = 0
        else:
            padding = 1
        if bn:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.Sigmoid())
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.Sigmoid())

    def forward(self, x):
        if self.is_circular:
            x = F.pad(x, pad=[0, self.kernel_size-1, 0, self.kernel_size-1] , mode='circular')
        out = self.conv(x)
        return out


def double_conv(in_channels, out_channels, bn=False, is_circular=False):
    return nn.Sequential(
        conv_block_unet(in_channels, out_channels, bn=bn, is_circular=is_circular),
        conv_block_unet(out_channels, out_channels, bn=bn, is_circular=is_circular)
    )


def last_conv(in_channels, out_channels, bn=False, is_circular=False):
    return nn.Sequential(
        conv_block_unet(in_channels, out_channels, bn=bn, is_circular=is_circular),
        conv_last_block_unet(out_channels, out_channels, bn=bn, is_circular=is_circular)
    )


class UNet(nn.Module):

    def __init__(self, in_channels, bn=False, is_circular=False):
        super().__init__()
                 
        self.dconv_down1 = double_conv(in_channels, 64, bn=bn, is_circular=is_circular)
        self.dconv_down2 = double_conv(64, 128, bn=bn, is_circular=is_circular)
        self.dconv_down3 = double_conv(128, 256, bn=bn, is_circular=is_circular)
        self.dconv_down4 = double_conv(256, 512, bn=bn, is_circular=is_circular)
        
        self.maxpool = nn.MaxPool2d(2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=128)
        self.avgpool1d = nn.AvgPool1d(kernel_size=128)    
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        # self.STE = StraightThroughEstimator()

        self.dconv_up3 = double_conv(256 + 512, 256, bn=bn, is_circular=is_circular)
        self.dconv_up2 = double_conv(128 + 256, 128, bn=bn, is_circular=is_circular)
        self.dconv_up1 = last_conv(128 + 64, 64, bn=bn, is_circular=is_circular)
        
        self.conv_last = conv2d(64, 1, 1, 1, 0, bn=bn, is_circular=is_circular)

    # check whether the vector is nan or not
    def check_tensor(self, vector):
        if isinstance(vector, torch.Tensor):
            if np.any(np.isnan(vector.cpu().detach().numpy())):
                print("The tensor has NaN")
                return True
        elif isinstance(vector, np.ndarray):
            if np.any(np.isnan(vector)):
                print("The array has NaN")
                return True
        elif isinstance(vector, list):
            vector = np.array(vector)
            if np.any(np.isnan(vector)):
                print("The list has NaN")
                return True

    def forward(self, x):
        batchsize = x.shape[0]  
        conv1 = self.dconv_down1(x)    
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)       
        x = self.maxpool(conv2)  
        conv3 = self.dconv_down3(x)        
        x = self.maxpool(conv3)      
        x = self.dconv_down4(x)
        x = self.upsample(x)          
        x = torch.cat([x, conv3], dim=1)     
        x = self.dconv_up3(x)    
        x = self.upsample(x)                  
        x = torch.cat([x, conv2], dim=1)       
        x = self.dconv_up2(x)     
        x = self.upsample(x)          
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)   
        out = self.conv_last(x) 
        # pool the output of the last layer
        # b, c, h, w = x.shape
        # x = x.reshape(-1, h, w)
        # # x = self.maxpool1d(x)
        # x = self.avgpool1d(x)
        # x = x.reshape(b, 1, c, h)
        # out = x.permute(0, 1, 3, 2)    
               
        # out = self.sigmoid(out)
        # out = torch.mul(x_copy,out)

        return out
# ------ UNet Architecture ------


# ------ STE Architecture ------
class _STE(torch.autograd.Function):
    """ Straight Through Estimator
    """

    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx,grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, input):
        # sign = apply(_STE, input)
        sign = _STE.apply
        output =  sign(input)
        return output
# ------ STE Architecture ------


# ------ Auto Encoder & Decoder Architecture ------
class conv_block_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, bn=False, is_circular=False):
        super(conv_block_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.is_circular = is_circular
        if is_circular:
            padding = 0
        else:
            padding = 1
        if bn:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.ReLU())

    def forward(self, x):
        if self.is_circular:
            x = F.pad(x, pad=[0, self.kernel_size-self.stride, 0, self.kernel_size-self.stride], mode='circular')
        out = self.conv(x)
        return out

class conv_block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bn=False):
        super(conv_block_decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if bn:
            self.conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())
        else:
            self.conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.ReLU())         

    def forward(self, x):
        out = self.conv(x)
        return out

class conv_last_block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bn=False):
        super(conv_last_block_decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if bn:
            self.conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.Tanh())
        else:
            self.conv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.Tanh())

    def forward(self, x):
        out = self.conv(x)
        return out

# use circular convolution to implement rotation invariance
class Autoencoder(nn.Module):
    def __init__(self, in_channels, bn=False, is_circular=False):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            conv_block_encoder(in_channels, 64, bn=bn, is_circular=is_circular), conv_block_encoder(64, 128, bn=bn, is_circular=is_circular), 
            conv_block_encoder(128, 256, bn=bn, is_circular=is_circular), conv_block_encoder(256, 512, bn=bn, is_circular=is_circular))

        self.decoder = nn.Sequential(
            conv_block_decoder(512, 256), conv_block_decoder(256, 128), 
            conv_block_decoder(128, 64), conv_last_block_decoder(64, 1))

        self.maxpool = nn.MaxPool1d(kernel_size=128)
        self.avgpool = nn.AvgPool1d(kernel_size=128)
        
    def forward(self, x, encode=False, decode=False):
        if encode:
            x = self.encoder(x)
        elif decode:
            x = self.decoder(x)
        else:
            encoding = self.encoder(x)
            x = self.decoder(encoding)
        return x        
# ------ Auto Encoder & Decoder Architecture ------
