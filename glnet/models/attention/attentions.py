import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet

EPS = 1e-4

from functools import partial

from glnet.ops.modules import MSDeformAttn, MSDeformAttn3D


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum
            

class L2Attention(nn.Module):
    """Compute the attention as L2-norm of local descriptors"""

    def forward(self, x):
        return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)


class VanillaSelfAttention(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim 
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=4, n_points=20)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, query_pos=None):
        '''
        query: (B, N, C)
        '''
        inp_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        device = query.device
        Z, X = 100, 100
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z-0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X-0.5, X, dtype=torch.float, device=query.device)
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2) # (B, N, 1, 2)

        B, N, C = query.shape
        input_spatial_shapes = query.new_zeros([1,2]).long()
        input_spatial_shapes[:] = 100
        input_level_start_index = query.new_zeros([1,]).long()
        queries = self.deformable_attention(query, reference_points, query.clone(), 
            input_spatial_shapes.detach(), input_level_start_index.detach())

        queries = self.output_proj(queries)

        return self.dropout(queries) + inp_residual


class SpatialCrossAttention(nn.Module):
    # From https://github.com/zhiqi-li/BEVFormer

    def __init__(self, dim=128, dropout=0.1):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim, num_heads=4, num_levels=1, num_points=20)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, query_pos=None, reference_points_cam=None, spatial_shapes=None, bev_mask=None, level_start_index=None):
        '''
        query: (B, N, C)
        key: (S, M, B, C)
        reference_points_cam: (S, B, N, D, 2), in 0-1
        bev_mask: (S, B, N, D)
        '''
        inp_residual = query.clone()
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        S, M, _, _ = key.shape

        D = reference_points_cam.size(3)
        indexes = []

        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros(
            [B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [B, S, max_len, D, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        key = key.permute(2, 0, 1, 3).reshape(
            B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(
            B * S, M, C)

        level_start_index = query.new_zeros([1,]).long()

        queries = self.deformable_attention(query=queries_rebatch.view(B*S, max_len, self.dim),
            key=key, value=value,
            reference_points=reference_points_rebatch.view(B*S, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        
        queries = queries.view(B, S, max_len, self.dim)

        for j in range(B):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0 
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


# Adafusion
class M_PosAttention2d(nn.Module):
    """Multi-head Position Attention 2D Module.
       Attention augmentation convolution, pytorch implementation of paper
    [1] Bello I, Zoph B, Vaswani A, et al. Attention augmented convolutional
        networks[C]//ICCV. 2019: 3286-3295.

        This class only contains the pure attention part for 2D inputs. The 
    combination of convolution should be done elsewhere outside this class.
        Currently no [relative] position encoding is available.

        Attention output channels: dv
    Args:
        dk, dv: Channels for K and V
        Nh: The number of heads
        dkh, dvh: Channels for K and V per head
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size,
        dk: int,
        dv: int,
        Nh: int,
        stride: int = 1,
        singleHead=False,
    ):
        """Note that out_channels = dv."""
        super(M_PosAttention2d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        # check parameters
        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."
        #
        self.dkh = self.dk // self.Nh
        self.dvh = self.dv // self.Nh

        # W_q, W_k, W_v as a whole matrix
        self.qkv_conv = nn.Conv2d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # W_O matrix
        self.attn_out = (
            nn.Identity()
            if singleHead
            else nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)
        )

    def forward(self, x):
        """Input x, shape (N, in_channels, H_ori, W_ori)"""

        # [Q,K,V] matrix, shape (N,2*dk+dv,H,W)
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()

        # shape (N, [dk, dk, dv], H, W)
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (self.dkh ** -0.5)  # the scale 1/sqrt(dkh) is multiplied to Q

        # split to multi-head, shape (N, Nh, dkh, H, W)
        q = torch.reshape(q, (N, self.Nh, self.dkh, H, W))

        # flatten Q, K or V. Combine (H,W) into (H*W,) shape
        # shape (N, Nh, dkh, H*W)
        flat_q = torch.reshape(q, (N, self.Nh, self.dkh, H * W))
        flat_k = torch.reshape(k, (N, self.Nh, self.dkh, H * W))
        flat_v = torch.reshape(v, (N, self.Nh, self.dvh, H * W))

        # logits = QK^T / sqrt(dkh), shape (N, Nh, H*W, H*W)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=-1)  # in [0, 1]

        # shape (N, Nh, H*W, dvh) -> (N, Nh, dvh, H*W)
        Oh = torch.matmul(weights, flat_v.transpose(2, 3))
        Oh = Oh.transpose(2, 3)

        # combine_heads O_all=[O1, O2, ... O_Nh], shape (N, dv, H, W)
        # attention out = O_all * W_O, shape (N, dv, H, W)
        O_all = torch.reshape(Oh, (N, self.dv, H, W))
        attn_out = self.attn_out(O_all)

        return attn_out


# Adafusion
class M_PosAttention3d(nn.Module):
    """Multi-head Position Attention 3D Module.
       Attention augmentation convolution, pytorch implementation of paper
    [1] Bello I, Zoph B, Vaswani A, et al. Attention augmented convolutional
        networks[C]//ICCV. 2019: 3286-3295.

        This class only contains the pure attention part for 3D inputs. The 
    combination of convolution should be done elsewhere outside this class.
        Currently no [relative] position encoding is available.

        Attention output channels: dv
    Args:
        dk, dv: Channels for K and V
        Nh: The number of heads
        dkh, dvh: Channels for K and V per head
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size,
        dk: int,
        dv: int,
        Nh: int,
        stride: int = 1,
        singleHead=False,
    ):
        """Note that out_channels = dv."""
        super(M_PosAttention3d, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        # check parameters
        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."
        #
        self.dkh = self.dk // self.Nh
        self.dvh = self.dv // self.Nh

        # W_q, W_k, W_v as a whole matrix
        self.qkv_conv = nn.Conv3d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # W_O matrix
        self.attn_out = (
            nn.Identity()
            if singleHead
            else nn.Conv3d(self.dv, self.dv, kernel_size=1, stride=1)
        )

    def forward(self, x):
        """Input x, shape (N, in_channels, X_ori, Y_ori, Z_ori)"""

        # [Q,K,V] matrix, shape (N, 2*dk+dv, X, Y, Z)
        qkv = self.qkv_conv(x)
        N, _, X, Y, Z = qkv.size()

        # shape (N, [dk, dk, dv], X, Y, Z)
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (self.dkh ** -0.5)  # the scale 1/sqrt(dkh) is multiplied to Q

        # split to multi-head, shape (N, Nh, dkh, X, Y, Z)
        q = torch.reshape(q, (N, self.Nh, self.dkh, X, Y, Z))

        # flatten Q, K or V. Combine (X,Y,Z) into (X*Y*Z,) shape
        # shape (N, Nh, dkh, X*Y*Z)
        flat_q = torch.reshape(q, (N, self.Nh, self.dkh, X * Y * Z))
        flat_k = torch.reshape(k, (N, self.Nh, self.dkh, X * Y * Z))
        flat_v = torch.reshape(v, (N, self.Nh, self.dvh, X * Y * Z))

        # logits = QK^T / sqrt(dkh), shape (N, Nh, X*Y*Z, X*Y*Z)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=-1)  # in [0, 1]

        # shape (N, Nh, X*Y*Z, dvh) -> (N, Nh, dvh, X*Y*Z)
        Oh = torch.matmul(weights, flat_v.transpose(2, 3))
        Oh = Oh.transpose(2, 3)

        # combine_heads O_all=[O1, O2, ... O_Nh], shape (N, dv, X, Y, Z)
        # attention out = O_all * W_O, shape (N, dv, X, Y, Z)
        O_all = torch.reshape(Oh, (N, self.dv, X, Y, Z))
        attn_out = self.attn_out(O_all)

        return attn_out


# Adafusion
class S_ChnAttention2d(nn.Module):
    """Single-head Channel Attention 2D Module.
       We implement the single head channel self-attention from paper
    [2] Fu, Jun, et al. "Dual attention network for scene segmentation." CVPR. 2019.

        This class only contains the pure attention part for 2D inputs. The 
    combination of convolution should be done elsewhere outside this class.
    """

    def __init__(self, in_chn: int, out_chn: int):
        """Args:
            in_chn: the input channal number
            out_chn: the desired output channal number"""
        super(S_ChnAttention2d, self).__init__()
        self.bottleneck = nn.Conv2d(in_chn, out_chn, kernel_size=1)

    def forward(self, x):
        """Input x, shape (N, C, H, W)"""

        N, C, H, W = x.size()  # shape (N, C, H, W)

        # combine H and W
        x_HW = torch.reshape(x, (N, C, H * W))  # shape (N, C, H*W)

        # A * A^T, is a symmetric matrix
        logits = torch.matmul(x_HW, x_HW.transpose(1, 2))  # shape (N, C, C)
        weights = F.softmax(logits, dim=-1)  # row in [0, 1], shape (N, C, C)

        # attention output
        attn_out = torch.matmul(weights, x_HW)  # shape (N, C, H*W)
        attn_out = torch.reshape(attn_out, (N, C, H, W))
        attn_out = self.bottleneck(attn_out)

        return attn_out


class S_ChnAttention3d(nn.Module):
    """Single-head Channel Attention 3D Module.
       We implement the single head channel self-attention from paper
    [2] Fu, Jun, et al. "Dual attention network for scene segmentation." CVPR. 2019.

        This class only contains the pure attention part for 3D inputs. The 
    combination of convolution should be done elsewhere outside this class.
    """

    def __init__(self, in_chn: int, out_chn: int):
        """Args:
            in_chn: the input channal number
            out_chn: the desired output channal number"""
        super(S_ChnAttention3d, self).__init__()
        self.bottleneck = nn.Conv3d(in_chn, out_chn, kernel_size=1)

    def forward(self, x):
        """Input x, shape (N, C, X, Y, Z)"""

        N, C, X, Y, Z = x.size()  # shape (N, C, X, Y, Z)

        # combine X, Y and Z
        x_XYZ = torch.reshape(x, (N, C, X * Y * Z))  # shape (N, C, X*Y*Z)

        # A * A^T, is a symmetric matrix
        logits = torch.matmul(x_XYZ, x_XYZ.transpose(1, 2))  # shape (N, C, C)
        weights = F.softmax(logits, dim=-1)  # row in [0, 1], shape (N, C, C)

        # attention output
        attn_out = torch.matmul(weights, x_XYZ)  # shape (N, C, X*Y*Z)
        attn_out = torch.reshape(attn_out, (N, C, X, Y, Z))
        attn_out = self.bottleneck(attn_out)

        return attn_out
