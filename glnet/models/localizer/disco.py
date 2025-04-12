from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from glnet.config.config import *
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from glnet.utils.params import ModelParams
import torchvision.transforms.functional as fn
from glnet.utils.data_utils.point_clouds import generate_bev, generate_bev_occ

###############
# DiSCO Network
###############

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )   

def last_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid()
    )


def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image, cmap='jet')
    plt.show()


class DiSCO(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(DiSCO, self).__init__()

        self.params = model_params
        self.dataset = model_params.dataset_type
        self.use_bev = model_params.use_bev
        self.use_normalize = model_params.normalize
        self.out_dim = model_params.output_dim
        self.theta = model_params.theta
        self.radius = model_params.radius
        self.col = int(np.sqrt(self.out_dim)/2)
        self.unet = UNet(20).cuda()
        if self.dataset == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif self.dataset == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        self.bounds = (pc_bev_conf['x_bound'][0], pc_bev_conf['x_bound'][1], pc_bev_conf['y_bound'][0], \
                       pc_bev_conf['y_bound'][1], pc_bev_conf['z_bound'][0], pc_bev_conf['z_bound'][1])
        self.X = pc_bev_conf['x_grid']
        self.Y = pc_bev_conf['y_grid']
        self.Z = pc_bev_conf['z_grid']
        self.device = torch.device("cuda:0")

    def forward_fft(self, input):
        median_output = torch.fft.fft2(input, dim=(-2, -1), norm="ortho")
        output = torch.sqrt(median_output.real**2 + median_output.imag**2 + 1e-15)
        output = torch.fft.fftshift(output)
        # output = fftshift2d(output)
        return output, median_output
    
    def forward(self, batch):
        pc = batch['pc'] # B,V,4
        orig_pc = batch['orig_pc'] # B,V,4
        
        if self.use_bev:
            bev = pc
        else:
            bev = generate_bev(orig_pc, Z=self.Z, Y=self.Y, X=self.X, bounds=self.bounds).unsqueeze(0).cuda()
            # bev = torch.from_numpy(generate_bev_occ(orig_pc, Z=self.Z, Y=self.Y, X=self.X, bounds=self.bounds)).unsqueeze(0).cuda()
        
        B, C, H, W = bev.shape
        
        # Polar transform
        out_h = self.radius
        out_w = self.theta
        new_h = torch.linspace(0, 1, out_h).view(-1, 1).repeat(1, out_w)
        new_w = torch.pi * torch.linspace(0, 2, out_w).repeat(out_h, 1)
        grid_xy = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
        new_grid = grid_xy.clone()
        new_grid[...,0] = grid_xy[...,0] * torch.cos(grid_xy[...,1])
        new_grid[...,1] = grid_xy[...,0] * torch.sin(grid_xy[...,1])
        new_grid = new_grid.unsqueeze(0).cuda().repeat(B,1,1,1)
        polar_bev = F.grid_sample(bev, new_grid, align_corners=False)
        
        unet_out = self.unet(polar_bev)
        del new_grid
        out, fft_result = self.forward_fft(unet_out)
        
        x = out.squeeze(1)
        x = x[:, (out_h//2 - self.col):(out_h//2 + self.col), (out_w//2 - self.col):(out_w//2 + self.col)]
        x = x.reshape(B, -1)
        if self.use_normalize:
            x = F.normalize(x, dim=1)
        
        return {'global': x, 'unet_out': unet_out, 'spec': out, 'fft_result': fft_result}
    
    def print_info(self):
        print('Model class: DiSCO')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))


class Corr2Softmax(nn.Module):
    def __init__(self, weight, bias):

        super(Corr2Softmax, self).__init__()
        softmax_w = torch.tensor((weight), requires_grad=True)
        softmax_b = torch.tensor((bias), requires_grad=True)
        self.softmax_w = torch.nn.Parameter(softmax_w)
        self.softmax_b = torch.nn.Parameter(softmax_b)
        self.register_parameter("softmax_w",self.softmax_w)
        self.register_parameter("softmax_b",self.softmax_b)
    def forward(self, x):
        x1 = self.softmax_w*x + self.softmax_b
        return x1

class UNet(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
                
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = last_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        
        
    def forward(self, x):
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
        
        return out

def phase_corr(a, b, device):
    # a: template; b: source
    # imshow(a.squeeze(0).float())
    # [B, 1, H, W, 2]
    eps = 1e-15

    real_a = a[...,0]
    real_b = b[...,0]
    imag_a = a[...,1]
    imag_b = b[...,1]

    # compute a * b.conjugate; shape=[B,H,W,C]
    R = torch.FloatTensor(a.shape[0], 1, a.shape[1], a.shape[2], 2).to(device)
    R[...,0] = real_a * real_b + imag_a * imag_b
    R[...,1] = real_a * imag_b - real_b * imag_a

    r0 = torch.sqrt(real_a ** 2 + imag_a ** 2 + eps) * torch.sqrt(real_b ** 2 + imag_b ** 2 + eps).to(device)
    R[...,0] = R[...,0].clone()/(r0 + eps).to(device)
    R[...,1] = R[...,1].clone()/(r0 + eps).to(device)

    corr = torch.ifft(R, 2)
    corr_real = corr[...,0]
    corr_imag = corr[...,1]
    corr = torch.sqrt(corr_real ** 2 + corr_imag ** 2 + eps)
    corr = fftshift2d(corr)

    corr_marginize = torch.sum(corr, 3, keepdim=False)
    angle = torch.max(corr_marginize)
 
    return angle, corr

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