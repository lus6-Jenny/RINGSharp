import os
import sys
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch_radon import Radon, ParallelBeam, RadonFanbeam

import glnet.utils.vox_utils.geom as geom
import glnet.utils.vox_utils.basic as basic
from glnet.utils.params import ModelParams
from glnet.config.config import *
from glnet.models.backbones_2d.base_lss_fpn import BaseLSSFPN
from glnet.models.backbones_2d.bev_depth_head import BEVDepthHead
from glnet.models.backbones_2d.unet import conv_block_unet, conv_last_block_unet, AdaptationBlock

EPS = 1e-4

from glnet.models.aggregation.GeM import GeM
from glnet.models.aggregation.NetVLADLoupe import NetVLADLoupe


class RINGSharpV(nn.Module):
    '''Modified from `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        yaw_mode (int: 0 or 1): 0 for applying convs before sinogram,
        1 for applying convs after sinogram.
            Default: True.
    '''

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, model_params: ModelParams, backbone_conf=backbone_conf):
        super(RINGSharpV, self).__init__()
        
        self.yaw_mode = 1 # 1
        self.trans_mode = 0 # 0

        if model_params.dataset_type == 'nclt':
            backbone_conf['final_dim'] = (224, 384)
        elif model_params.dataset_type == 'oxford':
            backbone_conf['final_dim'] = (320, 640)

        self.feature_dim = backbone_conf['output_channels']     
        self.output_dim = model_params.output_dim  
        self.use_submap = model_params.use_submap
        self.use_pretrained_model = model_params.use_pretrained_model
        
        self.theta = model_params.theta
        self.radius = model_params.radius
        self.coordinates = model_params.coordinates
        self.use_normalize = model_params.normalize
        self.aggregation = model_params.aggregation
        self.confidence = model_params.confidence
        
        self.encoder = BaseLSSFPN(**backbone_conf)
        
        if self.confidence:
            self.confidence_layer = AdaptationBlock(self.feature_dim, 1)
            
        self.image_meta_path = model_params.image_meta_path
        self.occ_conv_yaw = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.occ_conv_trans = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.pooling = NetVLADLoupe(feature_size=self.feature_dim, cluster_size=64,
        #                             output_dim=self.output_dim, gating=True, add_batch_norm=True)


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
        return spec, sinogram
    
    def forward(self, batch):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        '''
        pc = batch['pc']    # B,V,4
        im = batch['img']   # B,5,3,224,384 (B,S,C,H,W) S = number of cameras

        __p = lambda x: basic.pack_seqdim(x, B)
        __u = lambda x: basic.unpack_seqdim(x, B)

        # rgb_camXs = im
        # use the front 5 images
        num_views = 5
        rgb_camXs = im[:, :num_views, ...]
        B, S, C, H, W = rgb_camXs.shape
        
        assert(C==3)

        with open(os.path.expanduser(self.image_meta_path), 'rb') as handle:
            image_meta = pickle.load(handle)
        
        mats_dict = {}
        
        intrins = torch.from_numpy(np.array(image_meta['K'])).float()[:num_views, ...]
        pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
        cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:,:num_views,...]

        pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
        cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()
        body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
        pix_T_cams = pix_T_cams.view(B,1,S,4,4)
        cams_T_body = cams_T_body.view(B,1,S,4,4)
        body_T_cams = body_T_cams.view(B,1,S,4,4)
        ida_mats = torch.from_numpy(np.eye(4)).repeat(B*S,1,1).cuda().view(B,1,S,4,4)
        bda_mat = torch.from_numpy(np.eye(4)).repeat(B,1,1).cuda()

        mats_dict['sensor2ego_mats'] = body_T_cams.float()
        mats_dict['intrin_mats'] = pix_T_cams.float()
        mats_dict['ida_mats'] = ida_mats.float()
        mats_dict['bda_mat'] = bda_mat.float()

        # reshape tensors
        x = __p(rgb_camXs).view(B,1,S,C,H,W)
        x = x.float()

        # -------- View Transformation --------
        bev, depth_pred = self.encoder(x,
                                        mats_dict,
                                        timestamps=None,
                                        is_return_depth=True)
        B, C, H, W = bev.shape

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
            bev_yaw = self.occ_conv_yaw(bev)
            spec, sinogram = self.compute_spec(bev_yaw)
        elif self.yaw_mode == 1:
            # Apply conv layers after sinogram
            _, sinogram = self.compute_spec(bev)
            sinogram = self.occ_conv_yaw(sinogram)
            spec, fft_result = self.forward_row_fft(sinogram)
        else:
            # bev_yaw = torch.sum(bev, dim=-3).unsqueeze(-3)
            bev_yaw = bev
            spec, sinogram = self.compute_spec(bev_yaw)
        
        # -------- Translation BEV --------
        if self.trans_mode == 0:
            bev_trans = None
        elif self.trans_mode == 1:
            bev_trans = self.occ_conv_trans(bev)
        else:
            # bev_trans = torch.sum(bev, dim=-3).unsqueeze(-3)
            bev_trans = bev
        
        # -------- Global Descriptor  --------
        # # VLAD Aggregation
        # x = self.pooling(bev.permute(0,2,3,1).reshape(B, -1, C))
        x = torch.sum(spec, dim=-2).reshape(B, -1)
        if self.use_normalize:
            x = F.normalize(x, dim=-1)
        
        return {'bev': bev, 'depth': depth_pred, 'polar_bev': polar_bev, 'bev_trans': bev_trans, 'spec': spec, 'global': x, 'confidence': bev_confidence}
    

    def print_info(self):
        print('Model class: RING#-V')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.encoder.parameters()])
        print('Encoder parameters: {}'.format(n_params))