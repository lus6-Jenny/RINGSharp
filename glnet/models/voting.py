
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified from OrientNet

from typing import Optional, Tuple

import numpy as np
import torch
from torch.fft import irfftn, rfftn
from torch.nn.functional import grid_sample, log_softmax, pad, normalize
import torchvision.transforms.functional as fn


def angle_error(t, t_gt):
    error = torch.abs(t % 360 - t_gt.to(t) % 360)
    error = torch.minimum(error, 360 - error)
    return error


# @torch.jit.script
def make_grid(
    w: float,
    h: float,
    step_x: float = 1.0,
    step_y: float = 1.0,
    orig_x: float = 0,
    orig_y: float = 0,
    y_up: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    x, y = torch.meshgrid(
        [
            torch.arange(orig_x, w + orig_x, step_x, device=device),
            torch.arange(orig_y, h + orig_y, step_y, device=device),
        ],
        indexing="xy",
    )
    if y_up:
        y = y.flip(-2)
    grid = torch.stack((x, y), -1)
    return grid


# @torch.jit.script
def rotmat2d(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([c, -s, s, c], -1).reshape(angle.shape + (2, 2))
    return R


class TemplateSampler(torch.nn.Module):
    def __init__(self, num_rotations, optimize=False):
        super().__init__()
        if optimize:
            assert (num_rotations % 4) == 0
            angles = torch.arange(
                0, 90, 90 / (num_rotations // 4))
        else:
            angles = torch.arange(
                0, 360, 360 / num_rotations)
        rotmats = rotmat2d(angles / 180 * np.pi)

        self.optimize = optimize
        self.num_rots = num_rotations
        self.register_buffer("angles", angles, persistent=False)
        self.register_buffer("rotmats", rotmats, persistent=False)

    def forward(self, image_bev):
        if image_bev.dim() == 3:
            image_bev = image_bev.unsqueeze(0)
        b, c, h, w = image_bev.shape
        n = self.num_rots
        grid_xy = make_grid(
            w,
            h,
            orig_x=(1 - w) / 2,      
            orig_y=(1 - h) / 2,
            y_up=False,
        )         
        grid_norm = grid_xy / grid_xy.new_tensor([w, h]) * 2
        grid = torch.einsum("...nij,...hwj->...nhwi", self.rotmats, grid_norm)
        grid = grid[None].repeat_interleave(b, 0).reshape(b * n, h, w, 2)
        image = (
            image_bev[:, None]
            .repeat_interleave(n, 1)
            .reshape(b * n, *image_bev.shape[1:])
        )
        grid = grid.to(image.device)
        kernels = grid_sample(image, grid.to(image.dtype), align_corners=False).reshape(
            b, n, c, h, w
        )

        if self.optimize:  # we have computed only the first quadrant
            kernels_quad234 = [torch.rot90(kernels, -i, (-2, -1)) for i in (1, 2, 3)]
            kernels = torch.cat([kernels] + kernels_quad234, 1)

        return kernels

    def run(self, image_bev):
        b, c, h, w = image_bev.shape
        n = self.num_rots
        kernels = torch.zeros(b, n, c, h, w).to(image_bev.device)
        angles = np.arange(0, 360, 360 / n)
        for idx, angle in enumerate(angles):
            rotated_image_bev = fn.rotate(image_bev, angle, resample=0, expand=False)
            kernels[:,idx,:,:,:] = rotated_image_bev

        return kernels


def conv2d_fft_batchwise(signal, kernel):
    signal = normalize(signal, dim=(-1, -2))
    kernel = normalize(kernel, dim=(-1, -2))    
    signal_fr = rfftn(signal, dim=(-1, -2))
    kernel_fr = rfftn(kernel, dim=(-1, -2))

    kernel_fr.imag *= -1  # flip the kernel
    output_fr = torch.einsum("bc...,bdc...->bd...", signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=(-1, -2))

    return output


class SparseMapSampler(torch.nn.Module):
    def __init__(self, num_rotations):
        super().__init__()
        angles = torch.arange(0, 360, 360 / self.conf.num_rotations)
        rotmats = rotmat2d(angles / 180 * np.pi)
        self.num_rotations = num_rotations
        self.register_buffer("rotmats", rotmats, persistent=False)

    def forward(self, image_map, p2d_bev):
        h, w = image_map.shape[-2:]
        locations = make_grid(w, h, device=p2d_bev.device)
        p2d_candidates = torch.einsum(
            "kji,...i,->...kj", self.rotmats.to(p2d_bev), p2d_bev
        )
        p2d_candidates = p2d_candidates[..., None, None, :, :] + locations.unsqueeze(-1)
        # ... x N x W x H x K x 2

        p2d_norm = (p2d_candidates / (image_map.new_tensor([w, h]) - 1)) * 2 - 1
        valid = torch.all((p2d_norm >= -1) & (p2d_norm <= 1), -1)
        value = grid_sample(
            image_map, p2d_norm.flatten(-4, -2), align_corners=True, mode="bilinear"
        )
        value = value.reshape(image_map.shape[:2] + valid.shape[-4])
        return valid, value


def sample_xyr(volume, xyr):
    angle_index = torch.round(xyr[:, 2] * volume.shape[-1] / 360) # torch.ceil(xyr[:, 2] * volume.shape[-1] / 360) - 1
    x_index = torch.round(xyr[:, 0]) + volume.shape[-2] // 2 # torch.ceil(xyr[:, 0]) + volume.shape[-2] // 2
    y_index = torch.round(xyr[:, 1]) + volume.shape[-3] // 2 # torch.ceil(xyr[:, 1]) + volume.shape[-3] // 2
    value = torch.zeros_like(angle_index)
    for i in range(angle_index.shape[0]):
        if x_index[i] < 0 or x_index[i] >= volume.shape[-2] or y_index[i] < 0 or y_index[i] >= volume.shape[-3]:
            continue
        if angle_index[i] == volume.shape[-1]:
            # print(f'Angle index {angle_index[i]} is out of bounds!')
            angle_index[i] = volume.shape[-1] - 1
        value[i] = volume[i, x_index[i].long(), y_index[i].long(), angle_index[i].long()]
    return value


def nll_loss_xyr(log_probs, gt_xyr):
    log_prob = sample_xyr(log_probs, gt_xyr)
    nll = -log_prob.reshape(-1)  # remove C,H,W,N
    return nll


def nll_loss_xyr_smoothed(log_probs, xy, angle, sigma_xy, sigma_r, mask=None):
    *_, nx, ny, nr = log_probs.shape
    grid_x = torch.arange(nx, device=log_probs.device, dtype=torch.float)
    dx = (grid_x - xy[..., None, 0]) / sigma_xy
    grid_y = torch.arange(ny, device=log_probs.device, dtype=torch.float)
    dy = (grid_y - xy[..., None, 1]) / sigma_xy
    dr = (
        torch.arange(0, 360, 360 / nr, device=log_probs.device, dtype=torch.float)
        - angle[..., None]
    ) % 360
    dr = torch.minimum(dr, 360 - dr) / sigma_r
    diff = (
        dx[..., None, :, None] ** 2
        + dy[..., :, None, None] ** 2
        + dr[..., None, None, :] ** 2
    )
    pdf = torch.exp(-diff / 2)
    if mask is not None:
        pdf.masked_fill_(~mask[..., None], 0)
        log_probs = log_probs.masked_fill(~mask[..., None], 0)
    pdf /= pdf.sum((-1, -2, -3), keepdim=True)
    return -torch.sum(pdf * log_probs.to(torch.float), dim=(-1, -2, -3))


def log_softmax_spatial(x, dims=3):
    return log_softmax(x.flatten(-dims), dim=-1).reshape(x.shape)


# @torch.jit.script
def argmax_xy(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.flatten(-2).max(-1).indices
    width = scores.shape[-1]
    x = indices % width
    y = torch.div(indices, width, rounding_mode="floor")
    return torch.stack((x, y), -1)


# @torch.jit.script
def expectation_xy(prob: torch.Tensor) -> torch.Tensor:
    h, w = prob.shape[-2:]
    grid = make_grid(float(w), float(h), device=prob.device).to(prob)
    return torch.einsum("...hw,hwd->...d", prob, grid)


# @torch.jit.script
def expectation_xyr(
    prob: torch.Tensor, covariance: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    h, w, num_rotations = prob.shape[-3:]
    x, y = torch.meshgrid(
        [
            torch.arange(w, device=prob.device, dtype=prob.dtype) - w//2,
            torch.arange(h, device=prob.device, dtype=prob.dtype) - h//2,
        ],
        indexing="xy",
    )
    grid_xy = torch.stack((y, x), -1)
    xy_mean = torch.einsum("...hwn,hwd->...d", prob, grid_xy)

    angles = torch.arange(0, 1, 1 / num_rotations, device=prob.device, dtype=prob.dtype)
    angles = angles * 2 * np.pi
    grid_cs = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
    cs_mean = torch.einsum("...hwn,nd->...d", prob, grid_cs)
    angle = torch.atan2(cs_mean[..., 1], cs_mean[..., 0])
    angle = (angle * 180 / np.pi) % 360

    if covariance:
        xy_cov = torch.einsum("...hwn,...hwd,...hwk->...dk", prob, grid_xy, grid_xy)
        xy_cov = xy_cov - torch.einsum("...d,...k->...dk", xy_mean, xy_mean)
    else:
        xy_cov = None

    xyr_mean = torch.cat((xy_mean, angle.unsqueeze(-1)), -1)
    return xyr_mean, xy_cov


# @torch.jit.script
def argmax_xyr(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.flatten(-3).max(-1).indices
    values = scores.flatten(-3).max(-1).values
    height, width, num_rotations = scores.shape[-3:]    
    angle_index = indices % num_rotations
    xyr = torch.zeros((angle_index.shape[0], 3))
    for i in range(angle_index.shape[0]):
        xy_scores = scores[i, :, :, angle_index[i]].squeeze(-1)
        indices = xy_scores.flatten(-2).max(-1).indices
        x = torch.div(indices, height, rounding_mode="floor") - height//2
        y = indices % height - width//2
        angle = angle_index[i] * 360 / num_rotations
        xyr[i, 0] = x
        xyr[i, 1] = y
        xyr[i, 2] = angle

    return xyr


# @torch.jit.script
def mask_yaw_prior(
    scores: torch.Tensor, yaw_prior: torch.Tensor, num_rotations: int
) -> torch.Tensor:
    step = 360 / num_rotations
    step_2 = step / 2
    angles = torch.arange(step_2, 360 + step_2, step, device=scores.device)
    yaw_init, yaw_range = yaw_prior.chunk(2, dim=-1)
    rot_mask = angle_error(angles, yaw_init) < yaw_range
    return scores.masked_fill_(~rot_mask[:, None, None], -np.inf)


def fuse_gps(log_prob, uv_gps, ppm, sigma=10, gaussian=False):
    grid = make_grid(*log_prob.shape[-3:-1][::-1]).to(log_prob)
    dist = torch.sum((grid - uv_gps) ** 2, -1)
    sigma_pixel = sigma * ppm
    if gaussian:
        gps_log_prob = -1 / 2 * dist / sigma_pixel**2
    else:
        gps_log_prob = torch.where(dist < sigma_pixel**2, 1, -np.inf)
    log_prob_fused = log_softmax_spatial(log_prob + gps_log_prob.unsqueeze(-1))
    return log_prob_fused