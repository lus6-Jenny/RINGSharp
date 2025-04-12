import copy
import os
import torch
import numpy as np
import open3d as o3d
import pygicp
import glnet.utils.vox_utils.vox as vox
from glnet.utils.data_utils.extract_local_descriptor import build_neighbors_NN, get_pointfeat
from glnet.config.config import *
import matplotlib.pyplot as plt


def visualize_2d_data(data, save_path):
    # Convert data to NumPy array if it's a tensor
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    data = data.squeeze()
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the data
    ax.imshow(data, cmap='jet')
    # Add colorbar
    cbar = ax.figure.colorbar(ax.imshow(data, cmap='jet'))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def generate_bev_occ(pc, Z=20, Y=160, X=160, bounds=(-70.0, 70.0, -70.0, 70.0, -20.0, 20.0)):
    import voxelocc
    '''
    pc: original point cloud
    '''
    # returns Z * Y * X matrix
    pc = np.asarray(pc, dtype=np.float32)
    pc = pc[:,:3]
    # normalization
    mask = (pc[:,0] > bounds[0]) * (pc[:,0] < bounds[1]) * (pc[:,1] > bounds[2]) * \
        (pc[:,1] < bounds[3]) * (pc[:,2] > bounds[4]) * (pc[:,2] < bounds[5])
    pc = pc[mask]
    pc[:,0] = pc[:,0] * 2 / (bounds[1] - bounds[0])
    pc[:,1] = pc[:,1] * 2 / (bounds[3] - bounds[2])
    pc[:,2] = (pc[:,2] - bounds[4]) / (bounds[5] - bounds[4])
    size = pc.shape[0]
    pc_bev = np.zeros([Z * Y * X])
    pc = pc.transpose().flatten().astype(np.float32)
    
    transer_bev = voxelocc.GPUTransformer(pc, size, 1, 1, Y, X, Z, 1)
    transer_bev.transform()
    point_t_bev = transer_bev.retreive()
    point_t_bev = point_t_bev.reshape(-1, 3)
    point_t_bev = point_t_bev[...,2]

    pc_bev = point_t_bev.reshape(Z, Y, X)
    
    return pc_bev


def generate_feats(pc, Z=1, Y=160, X=160, bounds=(-70.0, 70.0, -70.0, 70.0, -20.0, 20.0)):
    '''
    pc: original point cloud
    '''
    pc = np.asarray(pc, dtype=np.float32)
    pc = pc[..., :3]
    
    # feature extraction
    k = 30  # num of neighbors
    pc_feats = get_pointfeat(pc, k)
    
    scene_centroid = [0.0, 0.0, 0.0]
    scene_centroid = torch.from_numpy(np.array(scene_centroid).reshape([1, 3])).float()
    vox_util = vox.Vox_util(Z, Y, X,
                            scene_centroid=scene_centroid,
                            bounds=bounds,
                            assert_cube=False)
    if not isinstance(pc, torch.Tensor):
        pc =  torch.tensor(pc).float()
    if not isinstance(pc_feats, torch.Tensor):
        pc_feats =  torch.tensor(pc_feats).float()
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
        pc_feats = pc_feats.unsqueeze(0)
    
    if torch.isnan(pc_feats).any():
        print('\n ------ Nan in point cloud features ------ \n')
        pc_feats = torch.where(torch.isnan(pc_feats), torch.tensor(0.0), pc_feats)
    B, N, C = pc_feats.shape
    feats = vox_util.voxelize_xyz_and_feats(pc, pc_feats, Z, Y, X, assert_cube=False)
    if Z == 1:
        feats = feats.reshape(B, C, Y, X)
    else:
        feats = feats.reshape(B, C, Z, Y, X)
    feats = feats.permute(*range(feats.dim()-2), -1, -2)
    
    return feats
    

def generate_bev(pc, Z=20, Y=160, X=160, bounds=(-70.0, 70.0, -70.0, 70.0, -20.0, 20.0)):
    '''
    pc: original point cloud
    '''
    # returns B * Z * Y * X matrix
    pc = pc[..., :3]
    scene_centroid = [0.0, 0.0, 0.0]
    scene_centroid = torch.from_numpy(np.array(scene_centroid).reshape([1, 3])).float()
    vox_util = vox.Vox_util(Z, Y, X,
                            scene_centroid=scene_centroid,
                            bounds=bounds,
                            assert_cube=False)
    if not isinstance(pc, torch.Tensor):
        pc =  torch.tensor(pc).float()
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
    occ_mem = vox_util.voxelize_xyz(pc, Z, Y, X, assert_cube=False)
    pc_bev = occ_mem.reshape(-1, Z, Y, X).squeeze(0)
    
    pc_bev = pc_bev.permute(*range(pc_bev.dim()-2), -1, -2)
    
    return pc_bev


def generate_bev_feats(pc, pc_feats, Z=1, Y=160, X=160, bounds=(-70.0, 70.0, -70.0, 70.0, -20.0, 20.0)):
    '''
    pc: original point cloud
    '''
    # returns B * C * Z * Y * X matrix
    pc = pc[...,:3]
    scene_centroid = [0.0, 0.0, 0.0]
    scene_centroid = torch.from_numpy(np.array(scene_centroid).reshape([1, 3])).float()
    vox_util = vox.Vox_util(Z, Y, X,
                            scene_centroid=scene_centroid,
                            bounds=bounds,
                            assert_cube=False)
    if not isinstance(pc, torch.Tensor):
        pc =  torch.tensor(pc).float()
    if not isinstance(pc_feats, torch.Tensor):
        pc_feats =  torch.tensor(pc_feats).float()  
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
        pc_feats = pc_feats.unsqueeze(0)
    B, N, C = pc_feats.shape
    feats = vox_util.voxelize_xyz_and_feats(pc, pc_feats, Z, Y, X, assert_cube=False)
    feats = feats.reshape(B, C, Z, Y, X)
    feats = feats.permute(*range(feats.dim()-2), -1, -2)
    
    return feats
    

# fast_gicp (https://github.com/SMRT-AIST/fast_gicp)
def fast_gicp(source, target, max_correspondence_distance=1.0, init_pose=np.eye(4)):
    # downsample the point cloud before registration
    source = pygicp.downsample(source, 0.25)
    target = pygicp.downsample(target, 0.25)

    # pygicp.FastGICP has more or less the same interfaces as the C++ version
    gicp = pygicp.FastGICP()
    gicp.set_input_target(target)
    gicp.set_input_source(source)

    # optional arguments
    gicp.set_num_threads(4)
    gicp.set_max_correspondence_distance(max_correspondence_distance)

    # align the point cloud using the initial pose calculated by RING
    T_matrix = gicp.align(initial_guess=init_pose)

    # get the fitness score
    fitness = gicp.get_fitness_score(max_range=1.0)
    # get the transformation matrix
    T_matrix = gicp.get_final_transformation()

    return fitness, T_matrix


# open3d icp
def o3d_icp(source, target, transform: np.ndarray = None, point2plane: bool = False,
        inlier_dist_threshold: float = 1.0, max_iteration: int = 200):
    # transform: initial alignment transform
    if transform is not None:
        transform = transform.astype(float)

    voxel_size = 0.25
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(source)
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(target)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    if point2plane:
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    if transform is not None:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold, transform,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p.fitness, reg_p2p.transformation, reg_p2p.inlier_rmse

    
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def draw_pc(pc):
    pc = copy.deepcopy(pc)
    pc.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([pc],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def icp(anchor_pc, positive_pc, transform: np.ndarray = None, point2plane: bool = False,
        inlier_dist_threshold: float = 1.5, max_iteration: int = 100):
    # transform: initial alignment transform
    if transform is not None:
        transform = transform.astype(float)

    voxel_size = 0.2
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(anchor_pc)
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positive_pc)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    if point2plane:
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        transform_estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    if transform is not None:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold, transform,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, inlier_dist_threshold,
                                                              estimation_method=transform_estimation,
                                                              criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse


def make_open3d_feature(data, dim, npts):
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


class PointCloudLoader:
    # Generic point cloud loader class
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = False
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname, sph=False, extrinsics_dir=None):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        if isinstance(file_pathname, str):
            assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
            pc, _ = self.read_pc(file_pathname, extrinsics_dir=extrinsics_dir)
        elif isinstance(file_pathname, list):
            for i in range(len(file_pathname)):
                assert os.path.exists(file_pathname[i]), f"Cannot open point cloud: {file_pathname[i]}"
            pc, _ = self.read_pc(file_pathname, extrinsics_dir=extrinsics_dir)

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]
        
        imgs = None

        return pc, imgs

    def read_pc(self, file_pathname, sph=False, extrinsics_dir=None):
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")


class PointCloudWithImageLoader:
    # Generic point cloud loader class
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = False
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname, sph=False, extrinsics_dir=None):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path

        if isinstance(file_pathname, str):
            assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
            pc, imgs = self.read_pcim(file_pathname, sph, extrinsics_dir)
            if self.remove_zero_points:
                mask = np.all(np.isclose(pc, 0), axis=1)
                pc = pc[~mask]

            if self.remove_ground_plane:
                mask = pc[:, 2] > self.ground_plane_level
                pc = pc[mask]
        elif isinstance(file_pathname, list):
            for i in range(len(file_pathname)): 
                assert os.path.exists(file_pathname[i]), f"Cannot open point cloud: {file_pathname[i]}"
                pc, imgs = self.read_pcim(file_pathname, sph, extrinsics_dir)

                if self.remove_zero_points:
                    mask = np.all(np.isclose(pc, 0), axis=1)
                    pc = pc[~mask]

                if self.remove_ground_plane:
                    mask = pc[:, 2] > self.ground_plane_level
                    pc = pc[mask]
        else:
            raise NotImplementedError("read_pcim only support str and list input")

        return pc, imgs

    def read_pcim(self, file_pathname, sph=False, extrinsics_dir=None):
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pcim must be overloaded in an inheriting class")
