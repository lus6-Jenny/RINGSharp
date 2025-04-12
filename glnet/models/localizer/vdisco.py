import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pickle

import glnet.utils.vox_utils.vox as vox
import glnet.utils.vox_utils.geom as geom
import glnet.utils.vox_utils.basic as basic
from glnet.models.attention.attentions import SpatialCrossAttention, VanillaSelfAttention
from glnet.utils.params import ModelParams

from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

EPS = 1e-4

from functools import partial
from einops.layers.torch import Rearrange, Reduce

from glnet.models.aggregation.GeM import GeM
from glnet.models.aggregation.NetVLADLoupe import NetVLADLoupe


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift2d(x):
    for dim in range(1, len(x.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift = n_shift + 1  # for odd-sized images
        x = roll_n(x, axis=dim, n=n_shift)
    return x  # last dim=2 (real&imag)


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip


import torchvision
class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x

# no radar/lidar integration
class vDiSCO(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(vDiSCO, self).__init__()

        self.Z, self.Y, self.X = model_params.Z, model_params.Y, model_params.X
        self.feature_dim = model_params.feature_dim
        self.output_dim = model_params.output_dim
        self.encoder_type = model_params.image_encoder
        self.theta = model_params.theta
        self.radius = model_params.radius
        self.use_normalize = model_params.normalize
        self.aggregation = model_params.aggregation

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        
        # Encoder
        self.feat2d_dim = feat2d_dim = feature_dim = self.feature_dim
        if self.encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif self.encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif self.encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEVFormer self & cross attention layers
        self.bev_queries = nn.Parameter(0.1*torch.randn(feature_dim, self.Z, self.X)) # C, Z, X
        self.bev_queries_pos = nn.Parameter(0.1*torch.randn(feature_dim, self.Z, self.X)) # C, Z, X
        num_layers = 6
        self.num_layers = num_layers
        self.self_attn_layers = nn.ModuleList([
            VanillaSelfAttention(dim=feature_dim) for _ in range(num_layers)
        ]) # deformable self attention
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(dim=feature_dim) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers)
        ])
        ffn_dim = 1028
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, feature_dim)) for _ in range(num_layers)
        ])
        self.norm3_layers = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers)
        ])

        self.vox_util = vox.Vox_util(self.Z, self.Y, self.X,
                                        scene_centroid=model_params.scene_centroid.cuda(),
                                        bounds=model_params.bounds,
                                        assert_cube=False)

        self.image_meta_path = os.path.expanduser(model_params.image_meta_path)
        self.conv = nn.Conv2d(in_channels=self.feature_dim, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.gem_conv = nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1, stride=1)

        # aggregation
        if self.aggregation == 'gem':
            self.pooling = GeM()
        elif self.aggregation == 'vlad':
            self.pooling = NetVLADLoupe(feature_size=256, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=True)
        elif self.aggregation == 'fft':
            pass
        elif self.aggregation == 'max':
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError('Unknown aggregation method: {}'.format(self.aggregation))


    def forward_fft(self, input):
        median_output = torch.fft.fft2(input, norm="ortho")
        output = torch.sqrt(median_output.real ** 2 + median_output.imag ** 2 + 1e-15)
        output = fftshift2d(output)
        return output, median_output


    def forward(self, batch):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        '''
        pc = batch['pc']    # B,V,4 
        im = batch['img']   # B,5,3,224,384 (B,S,C,H,W) S = number of cameras

        __p = lambda x: basic.pack_seqdim(x, B)
        __u = lambda x: basic.unpack_seqdim(x, B)

        rgb_camXs = im
        B, S, C, H, W = rgb_camXs.shape 

        B0 = B*S
        assert(C==3)

        with open(self.image_meta_path, 'rb') as handle:
            image_meta = pickle.load(handle)

        intrins = torch.from_numpy(np.array(image_meta['K'])).float()
        pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
        cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()
        
        pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
        cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()

        body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
        cam0_T_camXs = __p(geom.get_camM_T_camXs(body_T_cams, ind=0))

        pix_T_cams_ = __p(pix_T_cams)
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs).cuda()

        # reshape tensors
        rgb_camXs_ = __p(rgb_camXs)

        # rgb encoder
        device = rgb_camXs_.device

        feat_camXs_ = self.encoder(rgb_camXs_)
        # print(feat_camXs_.shape)

        _, C, Hf, Wf = feat_camXs_.shape
        feat_camXs = __u(feat_camXs_) # (B, S, C, Hf, Wf)
        # feat_clone = feat_camXs[0,...].clone()
        # feat_camXs[1,2:4,...] = feat_clone[:2,...]
        # feat_camXs[1,:2,...] = feat_clone[2:4,...]

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X
        
        # compute the image locations (no flipping for now)
        xyz_mem_ = basic.gridcloud3d(B0, Z, Y, X, norm=False, device=rgb_camXs.device) # B0, Z*Y*X, 3
        xyz_cam0_ = self.vox_util.Mem2Ref(xyz_mem_, Z, Y, X, assert_cube=False)
        xyz_camXs_ = geom.apply_4x4(camXs_T_cam0_, xyz_cam0_)
        xy_camXs_ = geom.camera2pixels(xyz_camXs_, pix_T_cams_) # B0, N, 2
        xy_camXs = __u(xy_camXs_) # B, S, N, 2, where N=Z*Y*X
        reference_points_cam = xy_camXs_.reshape(B, S, Z, Y, X, 2).permute(1, 0, 2, 4, 3, 5).reshape(S, B, Z*X, Y, 2)
        reference_points_cam[..., 0:1] = reference_points_cam[..., 0:1] / float(W)
        reference_points_cam[..., 1:2] = reference_points_cam[..., 1:2] / float(H)
        bev_mask = ((reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0)).squeeze(-1)

        # self & cross attentions
        bev_queries = self.bev_queries.clone().unsqueeze(0).repeat(B,1,1,1).reshape(B, self.feature_dim, -1).permute(0,2,1) # B, Z*X, C
        bev_queries_pos = self.bev_queries_pos.clone().unsqueeze(0).repeat(B,1,1,1).reshape(B, self.feature_dim, -1).permute(0,2,1) # B, Z*X, C
        bev_keys = feat_camXs.reshape(B, S, C, Hf*Wf).permute(1, 3, 0, 2) # S, M, B, C
        spatial_shapes = bev_queries.new_zeros([1, 2]).long()
        spatial_shapes[0, 0] = Hf
        spatial_shapes[0, 1] = Wf

        for i in range(self.num_layers):
            # self attention within the features (B, N, C)
            bev_queries = self.self_attn_layers[i](bev_queries, bev_queries_pos)

            # normalize (B, N, C)
            bev_queries = self.norm1_layers[i](bev_queries)

            # cross attention into the images
            bev_queries = self.cross_attn_layers[i](bev_queries, bev_keys, bev_keys, 
                query_pos=bev_queries_pos,
                reference_points_cam = reference_points_cam,
                spatial_shapes = spatial_shapes, 
                bev_mask = bev_mask)

            # normalize (B, N, C)
            bev_queries = self.norm2_layers[i](bev_queries)

            # feedforward layer (B, N, C)
            bev_queries = bev_queries + self.ffn_layers[i](bev_queries)

            # normalize (B, N, C)
            bev_queries = self.norm3_layers[i](bev_queries)

        feat_bev = bev_queries.permute(0, 2, 1).reshape(B, self.feature_dim, self.Z, self.X)
        out_h = self.radius
        out_w = self.theta
        new_h = torch.linspace(0, 1, out_h).view(-1, 1).repeat(1, out_w)
        new_w = torch.pi * torch.linspace(0, 2, out_w).repeat(out_h, 1)

        eps = 1e-8
        grid_xy = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
        new_grid = grid_xy.clone()
        new_grid[...,0] = grid_xy[...,0] * torch.cos(grid_xy[...,1])
        new_grid[...,1] = grid_xy[...,0] * torch.sin(grid_xy[...,1])
        new_grid = new_grid.unsqueeze(0).cuda().repeat(B,1,1,1)
        polar_img_bev = F.grid_sample(feat_bev, new_grid, align_corners=False)
        x = polar_img_bev
        B, C, rho, theta = x.shape

        if self.aggregation == 'vlad':
            x = x.permute(0,2,3,1)
            x = x.reshape(B, -1, C)
            x = self.pooling(x)
            x = self.fc_out(x)
        elif self.aggregation == 'fft':
            unet_out = self.conv(x)
            # plt.figure()
            # plt.imshow(x[0,0,...].detach().cpu().numpy(),cmap=plt.cm.jet)
            # plt.show()
            # plt.savefig('./bev-query.png')
            # plt.figure()
            # plt.imshow(x[1,0,...].detach().cpu().numpy(),cmap=plt.cm.jet)
            # plt.show()
            # plt.savefig('./bev-pos.png')
            x, fourier_spectrum = self.forward_fft(unet_out)
 
            # plt.figure()
            # plt.imshow((torch.abs(F.normalize(x[1,0,...], dim=1)-F.normalize(x[0,0,...], dim=1))).detach().cpu().numpy())
            # plt.colorbar(orientation='horizontal')
            # plt.clim(0, 1) 
            # plt.show()
            # plt.savefig('./bev-diff.png')
            x = x[:,:, (rho//2 - 8):(rho//2 + 8), (theta//2 - 8):(theta//2 + 8)]
        elif self.aggregation == 'gem':
            # x = self.gem_conv(x)
            x = self.pooling(x)
        elif self.aggregation == 'max':
            x = self.pooling(x)

        # x = self.conv(polar_img_bev)        
        # x, fourier_spectrum = self.forward_fft(x)
        # x = x[:,:, (self.radius//2 - 8):(self.radius//2 + 8), (self.theta//2 - 8):(self.theta//2 + 8)]

        x = x.reshape(B, self.output_dim)

        if self.use_normalize:
            x = F.normalize(x, dim=1)

        # x is (batch_size, output_dim) tensor
        if self.aggregation == 'fft':
            return {'global': x, 'unet_out': unet_out, 'spec': fourier_spectrum}
        else:
            return {'global': x}


    def print_info(self):
        print('Model class: vDiSCO')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.encoder.parameters()])
        print('Image Backbone parameters: {}'.format(n_params))


