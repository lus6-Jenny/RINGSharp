import os
import sys
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch_radon import Radon, ParallelBeam, RadonFanbeam

from mmcv.ops import Voxelization
import glnet.utils.vox_utils.geom as geom
import glnet.utils.vox_utils.basic as basic
from glnet.utils.params import ModelParams
from glnet.utils.data_utils.point_clouds import generate_bev, generate_bev_feats
from glnet.config.config import *
from glnet.models.backbones_2d.unet import last_conv_block, UNet, Autoencoder, AdaptationBlock

EPS = 1e-4

from glnet.models.aggregation.GeM import GeM
from glnet.models.aggregation.NetVLADLoupe import NetVLADLoupe
from glnet.models.backbones_2d.steerable_cnn import SteerableCNN


class RINGSharpL(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(RINGSharpL, self).__init__()
        
        self.yaw_mode = 1
        self.trans_mode = 0

        self.params = model_params
        self.dataset = model_params.dataset_type
        self.use_bev = model_params.use_bev
        self.point_encoder = model_params.point_encoder
        self.bev_encoder = model_params.bev_encoder
        self.feature_dim = model_params.feature_dim
        self.output_dim = model_params.output_dim
        self.use_submap = model_params.use_submap

        self.theta = model_params.theta
        self.radius = model_params.radius
        self.coordinates = model_params.coordinates
        self.use_normalize = model_params.normalize
        self.aggregation = model_params.aggregation
        self.confidence = model_params.confidence
        
        if self.dataset == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif self.dataset == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        self.bounds = (pc_bev_conf['x_bound'][0], pc_bev_conf['x_bound'][1], pc_bev_conf['y_bound'][0], \
                       pc_bev_conf['y_bound'][1], pc_bev_conf['z_bound'][0], pc_bev_conf['z_bound'][1])
        self.X = pc_bev_conf['x_grid']
        self.Y = pc_bev_conf['y_grid']
        self.Z = pc_bev_conf['z_grid']
        
        self.encoder = SteerableCNN(self.Z, self.feature_dim, bn=False)

        if self.confidence:
            self.confidence_layer = AdaptationBlock(self.feature_dim, 1)
        
        if self.use_bev and self.encoder == nn.Identity():
            self.feature_dim = self.Z
        if self.bev_encoder == 'unet':
            self.encoder_yaw = UNet(self.feature_dim, bn=False, is_circular=False)
        elif self.bev_encoder == 'autoencoder':
            self.encoder_yaw = Autoencoder(self.feature_dim, bn=False, is_circular=False)
        else:
            mid_channels = model_params.feature_dim
            self.encoder_yaw = last_conv_block(self.feature_dim, mid_channels, bn=False)
            
    def forward_row_fft(self, input):
        median_output = torch.fft.fft2(input, dim=-1, norm='ortho')
        median_output_r = median_output.real
        median_output_i = median_output.imag
        output = torch.sqrt(median_output_r ** 2 + median_output_i ** 2 + 1e-15)
        return output, median_output

    def forward_column_fft(self, input):
        median_output = torch.fft.fft2(input, dim=-2, norm='ortho')
        median_output_r = median_output.real
        median_output_i = median_output.imag
        output = torch.sqrt(median_output_r ** 2 + median_output_i ** 2 + 1e-15)
        output = torch.fft.fftshift(output)
        return output, median_output

    def forward_fft(self, input):
        median_output = torch.fft.fft2(input, norm='ortho')
        output = torch.sqrt(median_output.real ** 2 + median_output.imag ** 2 + 1e-15)
        output = torch.fft.fftshift(output)
        return output, median_output
    
    def compute_spec(self, x, rot_ang=2*np.pi):
        # Radon Transform
        n_angles = x.shape[-2]
        image_size = x.shape[-1]
        angles = torch.FloatTensor(np.linspace(0, rot_ang, n_angles).astype(np.float32))
        radon = ParallelBeam(image_size, angles)   
        sinogram = radon.forward(x)   
        spec, fft_result = self.forward_row_fft(sinogram)
        # print('spec shape', spec.shape)
        return spec, sinogram
        
    def forward(self, batch):
        # import pdb; pdb.set_trace()
        pc = batch['pc']    # B,V,4 
        im = batch['img']   # B,5,3,224,384 (B,S,C,H,W) S = number of cameras
        
        # -------- BEV Generation --------
        bev = self.encoder(pc)
        B, C, H, W = bev.shape
        # print(f'bev shape: {bev.shape}')

        if self.confidence:
            bev_confidence = self.confidence_layer(bev).sigmoid()
            bev = bev * bev_confidence
        else:
            bev_confidence = None

        if self.coordinates == 'polar':
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
        else:
            polar_bev = None
        
        # -------- Yaw BEV --------
        if self.yaw_mode == 0:
            # Apply conv layers before sinogram
            bev_yaw = self.encoder_yaw(bev)
            spec, sinogram = self.compute_spec(bev_yaw)
        elif self.yaw_mode == 1:
            # Apply conv layers after sinogram
            _, sinogram = self.compute_spec(bev)
            sinogram = self.encoder_yaw(sinogram)
            spec, fft_result = self.forward_row_fft(sinogram) 
        else:
            raise ValueError('Wrong yaw mode!')
        
        # -------- Translation BEV --------
        if self.trans_mode == 0:
            bev_trans = None
        elif self.trans_mode == 1:
            bev_trans = bev
        else:
            raise ValueError('Wrong trans mode!')
        
        # -------- Global Descriptor  --------
        # RT + FFT Aggregation
        x = torch.sum(spec, dim=-2).reshape(B, -1)
        if self.use_normalize:
            x = F.normalize(x, dim=-1)
        
        return {'bev': bev, 'polar_bev': polar_bev, 'bev_trans': bev_trans, 'spec': spec, 'global': x, 'confidence': bev_confidence}
    
    
    def print_info(self):
        print('Model class: RING#-L')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
