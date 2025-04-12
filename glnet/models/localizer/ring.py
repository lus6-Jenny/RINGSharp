import os
import sys
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch_radon import Radon, ParallelBeam, RadonFanbeam

from glnet.utils.params import ModelParams
from glnet.utils.data_utils.point_clouds import generate_bev, generate_bev_occ, generate_bev_feats
from glnet.config.config import *

EPS = 1e-4

from glnet.models.aggregation.GeM import GeM
from glnet.models.aggregation.NetVLADLoupe import NetVLADLoupe


class RING(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(RING, self).__init__()
   
        self.params = model_params
        self.dataset = model_params.dataset_type
        self.use_bev = model_params.use_bev
        self.use_submap = model_params.use_submap

        self.theta = model_params.theta
        self.radius = model_params.radius
        self.coordinates = model_params.coordinates
        self.use_normalize = model_params.normalize
        
        if self.dataset == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif self.dataset == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        self.bounds = (pc_bev_conf['x_bound'][0], pc_bev_conf['x_bound'][1], pc_bev_conf['y_bound'][0], \
                       pc_bev_conf['y_bound'][1], pc_bev_conf['z_bound'][0], pc_bev_conf['z_bound'][1])
        # self.X = pc_bev_conf['x_grid']
        # self.Y = pc_bev_conf['y_grid']
        # self.Z = pc_bev_conf['z_grid']
        self.X = self.Y = 120
        self.Z = 1
    
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
        pc = batch['pc']    # B,V,4
        im = batch['img']   # B,5,3,224,384 (B,S,C,H,W) S = number of cameras
        orig_pc = batch['orig_pc']    # B,V,4
        
        # -------- BEV Generation --------
        if self.use_bev:
            bev = pc
        else:
            bev = generate_bev(orig_pc, Z=self.Z, Y=self.Y, X=self.X, bounds=self.bounds).unsqueeze(0).cuda()
            # bev = torch.from_numpy(generate_bev_occ(orig_pc, Z=self.Z, Y=self.Y, X=self.X, bounds=self.bounds)).unsqueeze(0).cuda()
        
        B, C, H, W = bev.shape
        # print(f'bev shape: {bev.shape}')

        _, sinogram = self.compute_spec(bev)
        spec, fft_result = self.forward_row_fft(sinogram)
        bev_trans = bev

        # -------- Global Descriptor  --------
        # RT + FFT Aggregation
        x = torch.sum(spec, dim=-2).reshape(B, -1)
        # x, fourier_spectrum = self.forward_column_fft(spec)
        # x = x[..., (x.shape[-2]//2 - 8):(x.shape[-2]//2 + 8), (x.shape[-1]//2 - 8):(x.shape[-1]//2 + 8)].reshape(B, -1)
        # print('embedding shape', x.shape)
        if self.use_normalize:
            x = F.normalize(x, dim=-1)

        return {'bev': bev, 'polar_bev': None, 'bev_trans': bev, 'spec': spec, 'global': x}
    

    def print_info(self):
        print('Model class: RING')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
