# Base dataset classes, inherited by dataset-specific classes
import os
import gc
import pickle
from typing import List, Dict
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as transforms

from glnet.config.config import *
from glnet.datasets.nclt.nclt_raw import NCLTPointCloudLoader, NCLTPointCloudWithImageLoader
from glnet.datasets.oxford.oxford_raw import OxfordPointCloudLoader, OxfordPointCloudWithImageLoader
from glnet.utils.data_utils.point_clouds import visualize_2d_data, generate_bev, PointCloudLoader, PointCloudWithImageLoader
from glnet.utils.data_utils.poses import m2ypr
from glnet.utils.common_utils import _ex, to_numpy, to_torch
from glnet.datasets.range_image import range_projection
from glnet.datasets.augmentation import RandomTransformation
from glnet.datasets.panorama import Stitcher, generate_sph_image
from glnet.datasets.nclt.project_vel_to_cam_rot import project_vel_to_cam, project_vel_to_cam_oxford
import matplotlib.pyplot as plt

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, pose: np, positives_poses: Dict[int, np.ndarray] = None, filepaths: list = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # pose: pose as 4x4 matrix
        # positives_poses: relative poses of positive examples refined using ICP
        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.pose = pose
        self.positives_poses = positives_poses
        self.filepaths = filepaths


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array, pose: np.array = None, filepaths: list = None):
        # position: x, y position in meters
        # pose: 6 DoF pose (as 4x4 pose matrix)
        assert position.shape == (2,)
        assert pose is None or pose.shape == (4, 4)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position
        self.pose = pose
        self.filepaths = filepaths

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position, self.pose, self.filepaths


class TrainingDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_type: str, query_filename: str, transform=None, set_transform=None, image_transform=None, params=None):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        print("dataset type ", dataset_type)        
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.image_transform = image_transform
        self.pc_transform = RandomTransformation()
        with open(self.query_filepath, 'rb') as f:
            self.queries: Dict[int, TrainingTuple] = pickle.load(f)
        print('{} queries in the dataset'.format(len(self)))
        if self.dataset_type == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif self.dataset_type == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        self.stitcher = Stitcher(dataset_type, dataset_path)
        self.bounds = (pc_bev_conf['x_bound'][0], pc_bev_conf['x_bound'][1], pc_bev_conf['y_bound'][0], \
                       pc_bev_conf['y_bound'][1], pc_bev_conf['z_bound'][0], pc_bev_conf['z_bound'][1])
        self.X = pc_bev_conf['x_grid']
        self.Y = pc_bev_conf['y_grid']
        self.Z = pc_bev_conf['z_grid']
        self.params = params
        if self.dataset_type == 'oxford':
            self.extrinsics_dir = _ex(os.path.join(dataset_path, 'extrinsics'))
        else:
            self.extrinsics_dir = None

        # pc_loader must be set in the inheriting class
        if self.params.model_params.use_rgb:
            self.pcim_loader = get_pointcloud_with_image_loader(self.dataset_type)
        else:
            self.pcim_loader = get_pointcloud_loader(self.dataset_type)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        if self.params.model_params.use_panorama:
            sph = True
        else:
            sph = False

        if self.dataset_type == 'oxford':
            query_pc, query_imgs = self.pcim_loader(self.queries[ndx].filepaths, sph=sph, extrinsics_dir=self.extrinsics_dir)
            file_pathname = self.queries[ndx].filepaths[0]
            query_bev_path = file_pathname.replace('velodyne_left', 'bev')
            query_bev_path = query_bev_path.replace('png', 'npy')
            query_bev_path = query_bev_path.replace('bin', 'npy')
        else:
            file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
            query_bev_path = file_pathname.replace('velodyne_sync', 'bev')
            query_bev_path = query_bev_path.replace('bin', 'npy')
            
            if self.params.model_params.use_submap:
                file_pathname = file_pathname.replace('velodyne_sync', 'submap')
                file_pathname = file_pathname.replace('bin', 'npy')
            
            query_pc, query_imgs = self.pcim_loader(file_pathname, sph=sph)
        
        query_pc = query_pc[:, :3]
        query_depth_path = query_bev_path.replace('bev', 'depth')
        query_range_image_path = query_bev_path.replace('bev', 'range_image')                   
        if self.params.model_params.use_depth:
            depth_folder = query_depth_path.strip(query_depth_path.split('/')[-1])
            os.makedirs(depth_folder, exist_ok=True)
            generate_depth = False
            if os.path.isfile(query_depth_path):
                try:
                    depth_maps = np.load(query_depth_path)
                except:
                    print(f'Error loading {query_depth_path}')
                    generate_depth = True
            else:
                generate_depth = True
            if generate_depth:
                S, H, W, C = np.array(query_imgs).shape
                depth_maps = np.zeros([S, H, W])
                for cam_num in range(S):
                    hits_pc = np.concatenate([query_pc, np.ones((query_pc.shape[0], 1), dtype=np.float32)], axis=1)
                    if self.dataset_type == 'oxford':
                        hits_image = project_vel_to_cam_oxford(hits_pc.T, cam_num)
                    else:
                        hits_image = project_vel_to_cam(hits_pc.T, cam_num+1)

                    x_im = hits_image[0, :] / hits_image[2, :]
                    y_im = hits_image[1, :] / hits_image[2, :]
                    z_im = hits_image[2, :]

                    idx_infront = (z_im > 0) & (x_im > 0) & (x_im < W) & (y_im > 0) & (y_im < H)

                    x_im = x_im[idx_infront]
                    y_im = y_im[idx_infront]
                    z_im = z_im[idx_infront]

                    x_im_int = x_im.astype(np.int32)
                    y_im_int = y_im.astype(np.int32)

                    depth_maps[cam_num, y_im_int, x_im_int] = z_im
                    
                    # ##### visualize the depth map for debug ######
                    # if ndx % 200 == 1 and cam_num == 1:
                    #     vis_im = query_imgs[cam_num]
                    #     plt.figure()
                    #     plt.imshow(vis_im)
                    #     z_draw = (z_im-np.min(z_im))/(np.max(z_im)-np.min(z_im))
                    #     # plt.scatter(x_im_int, y_im_int, c=z_draw, cmap='jet', alpha=0.1, s=1)
                    #     plt.scatter(x_im, y_im, c=z_im%20.0/20.0, cmap='jet', alpha=0.1, s=1)
                    #     plt.savefig('./proj_rect_'+str(ndx)+'_'+str(cam_num)+'.jpg')
                    #     plt.close()
                    #     depth_map = depth_maps[cam_num]
                    #     depth_map_vis = (depth_map-np.min(depth_map))/(np.max(depth_map)-np.min(depth_map))*255
                    #     plt.imshow(depth_map_vis)
                    #     plt.savefig('./depth_map_'+str(ndx)+'_'+str(cam_num)+'.jpg')
                
                np.save(query_depth_path, depth_maps)
                print(f'Generating {query_depth_path}')
            depth_maps = torch.from_numpy(depth_maps).float()

        # Ground truth pose
        pose = self.queries[ndx].pose
        
        # Point cloud augmentation
        if self.params.model_params.xyz_aug:
            query_pc, T_aug = self.pc_transform(query_pc)
            pose = pose @ np.linalg.inv(T_aug)
        
        query_yaw, pitch, roll = m2ypr(pose)
        
        if self.params.model_params.use_bev:
            bev_folder = query_bev_path.strip(query_bev_path.split('/')[-1])
            os.makedirs(bev_folder, exist_ok=True)
            if os.path.isfile(query_bev_path) and not self.params.model_params.xyz_aug:
                query_bev = np.load(query_bev_path)
                query_bev = to_torch(query_bev)
            else:
                query_bev = generate_bev(query_pc, Z=self.Z, Y=self.Y, X=self.X, bounds=self.bounds) # tensor output
        elif self.params.model_params.use_range_image:
            range_image_folder = query_range_image_path.strip(query_range_image_path.split('/')[-1])
            os.makedirs(range_image_folder, exist_ok=True)
            if os.path.isfile(query_range_image_path) and not self.params.model_params.xyz_aug:
                query_depth = np.load(query_range_image_path)
            else:
                if query_pc.shape[-1] == 3:
                    cloud = np.concatenate((query_pc, np.ones((len(query_pc), 1))), axis=1)
                else:
                    cloud = query_pc
                query_depth, _, _, _ = range_projection(cloud)
                # visualize_2d_data(query_depth, f'query_depth.jpg')
            query_depth = to_torch(query_depth).unsqueeze(0)
        
        query_pc_tensor = to_torch(query_pc)
        if self.transform is not None:
            query_pc_tensor = self.transform(query_pc_tensor)

        if self.params.model_params.use_rgb:
            if self.image_transform is not None:
                query_imgs = [self.image_transform(e) for e in query_imgs]
            else:
                t = [transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                transform = transforms.Compose(t)
                query_imgs = [transform(e) for e in query_imgs]
            
            query_imgs = torch.stack(query_imgs).float()

            if self.params.model_params.use_depth:
                return query_pc, query_pc_tensor, query_imgs, ndx, pose, depth_maps

        if self.params.model_params.use_bev:
            return query_pc, query_pc_tensor, query_imgs, ndx, pose, query_bev
        elif self.params.model_params.use_range_image:
            return  query_pc, query_pc_tensor, query_imgs, ndx, pose, query_depth
        
        return query_pc, query_pc_tensor, query_imgs, ndx, pose, query_yaw


    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            if len(e) == 5:
                self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], e[4]))
            else:
                self.query_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], None))

        self.map_set = []
        for e in map_l:
            if len(e) == 5:
                self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], e[4]))
            else:
                self.map_set.append(EvaluationTuple(e[0], e[1], e[2], e[3], None))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions

    def get_map_poses(self):
        # Get map positions as (N, 2) array
        poses = np.zeros((len(self.map_set), 4, 4), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            poses[ndx] = pos.pose
        return poses

    def get_query_poses(self):
        # Get query positions as (N, 2) array
        poses = np.zeros((len(self.query_set), 4, 4), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            poses[ndx] = pos.pose
        return poses

def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    if dataset_type == 'nclt':
        return NCLTPointCloudLoader()
    elif dataset_type == 'oxford':
        return OxfordPointCloudLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")


def get_pointcloud_with_image_loader(dataset_type) -> PointCloudWithImageLoader:
    if dataset_type == 'nclt':
        return NCLTPointCloudWithImageLoader()
    elif dataset_type == 'oxford':
        return OxfordPointCloudWithImageLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")

