import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as fn
from glnet.models.localizer.circorr2 import circorr2
from glnet.models.voting import rotmat2d, make_grid, expectation_xy
import matplotlib.pyplot as plt


def array2tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)
    # if type(x) == np.ndarray:
    #     x = torch.from_numpy(x)
    return x


def dict2array(x):
    x = np.array(list(x.items()), dtype=object)
    return x

    
def apply_transform(pc: torch.Tensor, m: torch.Tensor):
    # Apply 4x4 SE(3) transformation matrix on (N, 3) point cloud or 3x3 transformation on (N, 2) point cloud
    assert pc.ndim == 2
    # n_dim = pc.shape[1]
    n_dim = 3
    # assert n_dim == 2 or n_dim == 3
    assert m.shape == (n_dim + 1, n_dim + 1)
    # (m @ pc.t).t = pc @ m.t
    pc[:,:3] = pc[:,:3] @ m[:n_dim, :n_dim].transpose(1, 0) + m[:n_dim, -1]
    return pc


def relative_pose(m1, m2):
    # SE(3) pose is 4x4 matrix, such that
    # Pw = [R | T] @ [P]
    #      [0 | 1]   [1]
    # where Pw are coordinates in the world reference frame and P are coordinates in the camera frame
    # m1: coords in camera/lidar1 reference frame -> world coordinate frame
    # m2: coords in camera/lidar2 coords -> world coordinate frame
    # returns: relative pose of the first camera with respect to the second camera
    #          transformation matrix to convert coords in camera/lidar1 reference frame to coords in
    #          camera/lidar2 reference frame
    #
    m = np.linalg.inv(m2) @ m1
    # !!!!!!!!!! Fix for relative pose !!!!!!!!!!!!!
    # m[:3, 3] = -m[:3, 3]
    return m


def relative_pose_batch(m1, m2):
    """
    Calculate the relative pose between two sets of poses represented as transformation matrices.

    Args:
        m1 (torch.Tensor): Tensor of shape (n, 4, 4) representing the first set of poses.
        m2 (torch.Tensor): Tensor of shape (n, 4, 4) representing the second set of poses.

    """
    if isinstance(m1, np.ndarray):
        m1 = torch.from_numpy(m1).float()
    if isinstance(m2, np.ndarray):
        m2 = torch.from_numpy(m2).float()
    m = torch.matmul(m2.inverse(), m1)

    return m


def fast_corr(a, b):
    C, H, W = a.shape[-3:]
    a = fn.normalize(a, mean=a.mean(), std=a.std())
    b = fn.normalize(b, mean=b.mean(), std=b.std())
    a_fft = torch.fft.fft2(a, dim=-2, norm="ortho")
    b_fft = torch.fft.fft2(b, dim=-2, norm="ortho")
    corr = torch.fft.ifft2(a_fft*b_fft.conj(), dim=-2, norm="ortho")  
    corr = torch.sqrt(corr.real**2 + corr.imag**2)
    corr = torch.sum(corr, dim=-3) # add this line for multi feature channels
    corr = torch.sum(corr, dim=-1).view(-1, H) 
    
    corr = torch.fft.fftshift(corr, dim=-1)
    angle = H//2 - torch.argmax(corr, dim=-1)
    max_corr = torch.max(corr, dim=-1)[0]
    dist = 1 - max_corr/(0.15*a.shape[-3]*H*W)
    dist = dist.cpu().numpy()
    angle = angle.cpu().numpy()
    return dist, angle


def estimate_yaw(query_embedding, positive_embedding, scale=1):
    circorr = circorr2(is_circular=True, zero_mean_normalize=True)
    corr, score, angle = circorr(query_embedding, positive_embedding, scale)
    # dist = 1 - score
    
    return corr, score, angle


def angle_clip(angle):
    if angle > 0:
        if angle > np.pi:
            return angle - 2*np.pi
    else:
        if angle < -np.pi:
            return angle + 2*np.pi
    return angle


def rotate_bev(bev, angle):
    # convert angle to degree
    angle = angle * 180 / torch.pi
    bev_rotated = fn.rotate(bev, float(angle))

    return bev_rotated


def rotate_bev_batch(bev, angles: torch.Tensor):
    if angles.dim() == 0:
        angles = angles.unsqueeze(0)
    rotmats = rotmat2d(angles)
    if bev.dim() == 3:
        bev = bev.unsqueeze(0)
    B, C, H, W = bev.shape
    grid_xy = make_grid(
            W,
            H,
            orig_x=(1 - W) / 2,      
            orig_y=(1 - H) / 2,
            y_up=False,
        )  
    grid_norm = grid_xy / grid_xy.new_tensor([W, H]) * 2
    grid = torch.einsum("...nij,...hwj->...nhwi", rotmats.cuda(), grid_norm.cuda())
    N, H, W = grid.shape[:3]
    grid = grid[None].repeat_interleave(B, 0).reshape(B * N, H, W, 2)
    grid = grid.to(bev.device).to(bev.dtype)
    bev_batch = bev[:, None].repeat_interleave(N, 1).reshape(B * N, C, H, W)
    bev_rotated = F.grid_sample(bev_batch, grid, align_corners=False).reshape(B, N, C, H, W)

    return bev_rotated.squeeze(0)


def solve_translation(a, b, zero_mean_normalize=True, scale=1):
    h, w = a.shape[-2:]
    if zero_mean_normalize:
        # a = fn.normalize(a, mean=a.mean(), std=a.std())
        # b = fn.normalize(b, mean=b.mean(), std=b.std())
        InstanceNorm = nn.InstanceNorm2d(1, affine=False, track_running_stats=False) # ! better than fn.normalize when evaluating
        a = InstanceNorm(a)
        b = InstanceNorm(b)
    else:
        a = F.normalize(a, dim=(-2, -1))
        b = F.normalize(b, dim=(-2, -1))
    
    # 2D cross correlation
    a_fft = torch.fft.fft2(a, dim=(-2,-1), norm="ortho")
    b_fft = torch.fft.fft2(b, dim=(-2,-1), norm="ortho")
    corr = torch.fft.ifft2(a_fft*b_fft.conj(), dim=(-2,-1), norm="ortho")  
    corr = torch.sqrt(corr.real**2 + corr.imag**2 + 1e-15)
    corr = torch.sum(corr, dim=-3) # add this line for multi feature channels
    
    # shift the correlation to the center
    corr = torch.fft.fftshift(corr, dim=(-2, -1))
    
    # space interpolation
    if scale != 1:
        corr = F.interpolate(corr.unsqueeze(1), scale_factor=scale, mode="bilinear", align_corners=False).squeeze(1)

    if a.dim() == 4:
        x = torch.zeros(a.shape[0])
        y = torch.zeros(a.shape[0])
        errors = torch.zeros(a.shape[0])
        for i in range(a.shape[0]):
            corr_i = corr[i]
            idx_x = (corr_i == torch.max(corr_i)).nonzero()[0][0]
            idx_y = (corr_i == torch.max(corr_i)).nonzero()[0][1]
            x[i] = scale * h//2 - idx_x
            y[i] = scale * w//2 - idx_y
            errors[i] = 1 - torch.max(corr_i)/(0.15*h*w)
    else:
        idx_x = (corr==torch.max(corr)).nonzero()[0][0]
        idx_y = (corr==torch.max(corr)).nonzero()[0][1]        
        x = scale * h//2 - idx_x
        y = scale * w//2 - idx_y
        errors = 1 - torch.max(corr)/(0.15*h*w)

    return x.cpu().numpy(), y.cpu().numpy(), errors, corr


def phase_corr(a, b, corr2soft=None):
    # a: template; b: source
    # imshow(a.squeeze(0).float())
    # import pdb; pdb.set_trace()
    eps = 1e-15
    C, H, W = a.shape[-3:]
    a_fft = torch.fft.fft2(a, dim=(-2, -1), norm="ortho")
    b_fft = torch.fft.fft2(b, dim=(-2, -1), norm="ortho")
    corr = torch.fft.ifft2(a_fft*b_fft.conj(), dim=(-2, -1), norm="ortho")  
    corr = torch.sqrt(corr.real**2 + corr.imag**2 + eps)
    corr = torch.sum(corr, dim=-3) # add this line for multi feature channels
    corr = torch.sum(corr, dim=-2).view(-1, W)
    
    corr = torch.fft.fftshift(corr, dim=-1)
    angle = torch.argmax(corr, dim=-1)
    # angle = angle % W
    angle = angle - W//2

    return angle, corr
