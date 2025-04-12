import torch
import e2cnn.nn as nn
from e2cnn.nn import init
from e2cnn import gspaces


def conv7x7(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=3,
            dilation=1, bias=False):
    """7x7 convolution with padding"""
    return nn.R2Conv(in_type, out_type, 7,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv5x5(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=2,
            dilation=1, bias=False):
    """5x5 convolution with padding"""
    return nn.R2Conv(in_type, out_type, 5,
                      stride=stride,
                      padding=padding, 
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv3x3(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv1x1(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=0,
            dilation=1, bias=False):
    """1x1 convolution with padding"""
    return nn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv_block(in_type: nn.FieldType, out_type: nn.FieldType, kernel_size=3, bn=False):
    assert kernel_size in [1, 3, 5, 7], f'kernel_size must be in [1, 3, 5, 7]'
    if kernel_size == 1:
        conv = conv1x1
    elif kernel_size == 3:
        conv = conv3x3
    elif kernel_size == 5:
        conv = conv5x5
    elif kernel_size == 7:
        conv = conv7x7
    
    if bn:
        block = nn.SequentialModule(
                conv(in_type, out_type),
                nn.InnerBatchNorm(out_type),
                nn.ReLU(out_type, inplace=True)
            )
    else:
        block = nn.SequentialModule(
                conv(in_type, out_type),
                nn.ReLU(out_type, inplace=True)
            )
    
    return block
           

class SteerableCNN(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels=128, num_rotations=8, kernel_size=3, bn=False):
        
        super(SteerableCNN, self).__init__()
        
        self.r2_act = gspaces.Rot2dOnR2(N=num_rotations)
        self.in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * in_channels)
        self.out_type = nn.FieldType(self.r2_act, [self.r2_act.regular_repr] * out_channels)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = self.in_type
        out_type = nn.FieldType(self.r2_act, [self.r2_act.regular_repr] * 64)
        self.block1 = conv_block(in_type, out_type, kernel_size=kernel_size, bn=bn)
        
        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, [self.r2_act.regular_repr] * 64)
        self.block2 = conv_block(in_type, out_type, kernel_size=kernel_size, bn=bn)
        
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        in_type = self.block2.out_type
        out_type = nn.FieldType(self.r2_act, [self.r2_act.regular_repr] * 128)
        self.block3 = conv_block(in_type, out_type, kernel_size=kernel_size, bn=bn)
        
        in_type = self.block3.out_type
        out_type = self.out_type
        if kernel_size == 1:
            conv = conv1x1
        elif kernel_size == 3:
            conv = conv3x3
        elif kernel_size == 5:
            conv = conv5x5
        elif kernel_size == 7:
            conv = conv7x7
        self.last_conv = conv(in_type, out_type)
        # self.last_conv = conv5x5(in_type, out_type)
        
        self.gpool = nn.GroupPooling(out_type)
        
    
    def forward(self, input: torch.Tensor):
        # import pdb; pdb.set_trace()
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.in_type)
        
        # apply each equivariant block
        x = self.block1(x)
        x = self.block2(x)
        # x = self.pool1(x)
        x = self.block3(x)
        x = self.last_conv(x)
        
        # pool over the group
        x = self.gpool(x)
        
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        return x