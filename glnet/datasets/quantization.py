import numpy as np
from typing import List
from abc import ABC, abstractmethod
import torch
import MinkowskiEngine as ME
import os
import pickle
import glnet.utils.vox_utils.misc as misc
import glnet.utils.vox_utils.improc as improc
import glnet.utils.vox_utils.vox as vox
import glnet.utils.vox_utils.geom as geom
import glnet.utils.vox_utils.basic as basic


class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc):
        pass

    @abstractmethod
    def dequantize(self, coords):
        pass
    
    # @abstractmethod
    # def keypoint_position(self, supervoxel_centers, stride, kp_offset):
    #     pass


class EgoNNPolarQuantizer(Quantizer):
    def __init__(self, quant_step: List[float]):
        assert len(quant_step) == 3, '3 quantization steps expected: for sector (in degrees), ring and z-coordinate (in meters)'
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)
        self.theta_range = int(360. // self.quant_step[0])
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)

    def __call__(self, pc):
        # Convert to polar coordinates and quantize with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3

        # theta is an angle in degrees in 0..360 range
        theta = 180. + torch.atan2(pc[:, 1], pc[:, 0]) * 180./np.pi
        # dist is a distance from a coordinate origin
        dist = torch.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        z = pc[:, 2]
        polar_pc = torch.stack([theta, dist, z], dim=1)
        # Scale each coordinate so after quantization with step 1. we got the required quantization step in each dim
        polar_pc = polar_pc / self.quant_step
        quantized_polar_pc, ndx = ME.utils.sparse_quantize(polar_pc, quantization_size=1., return_index=True)
        # Return quantized coordinates and index of selected elements
        return quantized_polar_pc, ndx

    def to_cartesian(self, pc):
        # Convert to radian in -180..180 range
        theta = np.pi * (pc[:, 0] - 180.) / 180.
        x = torch.cos(theta) * pc[:, 1]
        y = torch.sin(theta) * pc[:, 1]
        z = pc[:, 2]
        cartesian_pc = torch.stack([x, y, z], dim=1)
        return cartesian_pc

    def dequantize(self, coords):
        # Dequantize coords and convert to cartesian as (N, 3) tensor of floats
        pc = (0.5 + coords) * self.quant_step.to(coords.device)
        return self.to_cartesian(pc)

    def keypoint_position(self, supervoxel_centres, stride, kp_offset):
        # Add voxel center position: 0.5 * self.voxel_size
        # to offset from the supervoxel centre value (in -1..1 range converted to absolute values):
        # self.voxel_size + features * super_voxel_size / 2
        device = supervoxel_centres.device
        supervoxel_centres = (supervoxel_centres + 0.5) * self.quant_step.to(device)
        supervoxel_size = torch.tensor(stride, dtype=torch.float, device=supervoxel_centres.device) * \
                          self.quant_step.to(device)
        #kp_pos = supervoxel_centres
        kp_pos = supervoxel_centres + kp_offset * supervoxel_size / 2.

        kp_pos = self.to_cartesian(kp_pos)
        return kp_pos


class PolarQuantizer(Quantizer):
    def __init__(self, image_meta_path: str, quant_step: List[float]):
        assert len(quant_step) == 3, '3 quantization steps expected: for sector (in degrees), ring and z-coordinate (in meters)'
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)
        self.image_meta_path = os.path.expanduser(image_meta_path)
        self.theta_range = int(360. // self.quant_step[0])

    def __call__(self, pc):
        # Convert to polar coordinates and quantize with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[-1] >= 3

        B, V, D = pc.shape
        __p = lambda x: basic.pack_seqdim(x, B)
        __u = lambda x: basic.unpack_seqdim(x, B)

        with open(self.image_meta_path, 'rb') as handle:
            image_meta = pickle.load(handle)

        intrins = torch.from_numpy(np.array(image_meta['K'])).float()
        pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
        cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()

        pix_T_cams = pix_T_cams.repeat(B,1,1,1)
        cams_T_body = cams_T_body.repeat(B,1,1,1)

        xyz_velo0 = pc
        # mag = torch.norm(xyz_velo0[...,:3], dim=2)
        # xyz_velo0 = xyz_velo0[:,mag[0]>1]
        
        xyz_cam0 = geom.apply_4x4(cams_T_body[:,0], xyz_velo0[...,:3])

        # theta is an angle in degrees in 0..360 range
        theta = 180. + torch.atan2(xyz_cam0[..., 0], xyz_cam0[..., 2]) * 180./np.pi
        # dist is a distance from a coordinate origin
        dist = torch.sqrt(xyz_cam0[..., 0]**2 + xyz_cam0[..., 2]**2)
        z = xyz_cam0[..., 1]
        if pc.shape[-1] > 3:
            xyz_extra = pc[..., 3:]
            xyz_extra = xyz_extra.squeeze(-1)
            polar_pc = torch.stack([dist, z, theta, xyz_extra], dim=-1)
            # Scale each coordinate so after quantization with step 1. we got the required quantization step in each dim
            polar_pc = polar_pc[..., :3] / self.quant_step
        else:
            polar_pc = torch.stack([dist, z, theta], dim=-1)
            # Scale each coordinate so after quantization with step 1. we got the required quantization step in each dim
            polar_pc = polar_pc / self.quant_step 

        return polar_pc

    def to_cartesian(self, pc):
        # Convert to radian in -180..180 range
        theta = np.pi * (pc[..., 2] - 180.) / 180.
        x = torch.cos(theta) * pc[..., 0]
        y = torch.sin(theta) * pc[..., 0]
        z = pc[..., 1]
        cartesian_pc = torch.stack([x, y, z], dim=-1)
        return cartesian_pc

    def dequantize(self, coords):
        # Dequantize coords and convert to cartesian as (N, 3) tensor of floats
        pc = coords * self.quant_step.to(coords.device)
        return self.to_cartesian(pc)


class CartesianQuantizer(Quantizer):
    def __init__(self, quant_step: float):
        self.quant_step = quant_step

    def __call__(self, pc):
        # Converts to polar coordinates and quantizes with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3
        quantized_pc, ndx = ME.utils.sparse_quantize(pc, quantization_size=self.quant_step, return_index=True)
        # Return quantized coordinates and index of selected elements
        return quantized_pc, ndx

    def dequantize(self, coords):
        # Dequantize coords and return as (N, 3) tensor of floats
        # Use coords of the voxel center
        pc = (0.5 + coords) * self.quant_step
        return pc

    def keypoint_position(self, supervoxel_centers, stride, kp_offset):
        # Add voxel center position: 0.5 * self.voxel_size
        # to offset from the supervoxel centre value (in -1..1 range converted to absolute values):
        # self.voxel_size + features * super_voxel_size / 2
        supervoxel_centres = (supervoxel_centers + 0.5) * self.quant_step
        supervoxel_size = torch.tensor(stride, dtype=torch.float, device=supervoxel_centres.device) * self.quant_step
        if kp_offset is not None:
            kp_pos = supervoxel_centres + kp_offset * supervoxel_size / 2.
        else:
            kp_pos = supervoxel_centres
        return kp_pos


class PassThroughQuantizer(Quantizer):
    def __init__(self):
        self.name = 'passthrough'

    def __call__(self, pc):
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        pc = pc.squeeze()
        assert pc.shape[1] == 3
        # Return original pointcloud
        return pc, None

    def dequantize(self, pc):
        return pc

    def keypoint_position(self, supervoxel_centers, stride, kp_offset):
        raise NotImplementedError(f"PassThrough not implement this interface")
    

if __name__ == "__main__":
    n = 1000
    cart = torch.rand((n, 3), dtype=torch.float)
    cart[:, 0] = cart[:, 0] * 200. - 100.
    cart[:, 1] = cart[:, 1] * 200. - 100.
    cart[:, 2] = cart[:, 2] * 30. - 10.

    quantizer = PolarQuantizer([0.5, 0.3, 0.2])
    polar_quant, ndx = quantizer(cart)
    back2cart = quantizer.dequantize(polar_quant)
    cart_filtered = cart[ndx]
    dist = torch.norm(back2cart - cart_filtered, dim=1)
    print(f'Residual error - min: {torch.min(dist):0.5f}   max: {torch.max(dist):0.5f}   mean: {torch.mean(dist):0.5f}')
