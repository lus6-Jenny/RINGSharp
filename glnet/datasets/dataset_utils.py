# Zhejiang University

import numpy as np
from typing import List
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from glnet.datasets.base_datasets import TrainingDataset, EvaluationTuple
from glnet.datasets.augmentation import TrainTransform, TrainSetTransform, TrainRGBTransform, ValRGBTransform
from glnet.datasets.samplers import BatchSampler
from glnet.utils.params import TrainingParams

def make_datasets(params: TrainingParams, validation=True):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.aug_mode)

    if params.model_params.use_rgb:
        image_train_transform = TrainRGBTransform(params.aug_mode)
        image_val_transform = ValRGBTransform()
    else:
        image_train_transform = None
        image_val_transform = None

    datasets['global_train'] = TrainingDataset(params.dataset_folder, params.dataset, params.train_file,
                                               transform=train_transform, set_transform=train_set_transform, 
                                               image_transform=image_train_transform, params=params)

    if validation:
        datasets['global_val'] = TrainingDataset(params.dataset_folder, params.dataset, params.val_file, image_transform=image_val_transform, params=params)

    return datasets


def make_collate_fn(dataset: TrainingDataset, params: TrainingParams):
    def collate_fn(data_list):

        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        clouds_tensor = [e[1] for e in data_list]
        images = [e[2] for e in data_list]
        labels = [e[3] for e in data_list]
        poses = [e[4] for e in data_list]        
        if params.model_params.use_depth:
            depth_maps = [e[5] for e in data_list]
            depth_maps = torch.stack(depth_maps)
        elif params.model_params.use_bev:
            bevs = [e[5] for e in data_list]
            bevs = torch.stack(bevs)
            depth_maps = None
        elif params.model_params.use_range_image:
            depths = [e[5] for e in data_list]
            depths = torch.stack(depths)
            depth_maps = None                       
        else:
            depth_maps = None
        
        B = len(clouds)

        if params.model_params.use_xyz:
            V = params.model_params.lidar_fix_num
            for i in range(B):
                lidar_data = clouds_tensor[i][:,:3]
                lidar_extra = clouds_tensor[i][:,3:]
                if clouds_tensor[i].shape[0] > V:
                    lidar_data = lidar_data[:V]
                    lidar_extra = lidar_extra[:V]
                elif clouds_tensor[i].shape[0] < V:
                    lidar_data = np.pad(lidar_data,[(0,V-lidar_data.shape[0]),(0,0)],mode='constant')
                    lidar_extra = np.pad(lidar_extra,[(0,V-lidar_extra.shape[0]),(0,0)],mode='constant',constant_values=-1)
                    lidar_data = torch.from_numpy(lidar_data).float()
                    lidar_extra = torch.from_numpy(lidar_extra).float()

                clouds_tensor[i] = torch.cat([lidar_data, lidar_extra], dim=1)

            # if dataset.set_transform is not None:
            #     # Apply the same transformation on all dataset elements
            #     lens = [len(cloud) for cloud in clouds_tensor]
            #     clouds_tensor = torch.cat(clouds_tensor, dim=0)
            #     clouds_tensor = dataset.set_transform(clouds_tensor)
            #     clouds_tensor = clouds_tensor.split(lens)
            
            clouds_tensor = torch.stack(clouds_tensor, dim=0)
            if params.model_params.quantizer == None:
                clouds_quant = clouds_tensor
            else:
                clouds_quant = params.model_params.quantizer(clouds_tensor)
        else:
            clouds_quant = None

        if params.model_params.use_rgb:
            images = torch.cat(images, dim=0)
            images = images.reshape(B, -1, images.shape[1], images.shape[2], images.shape[3])
            images = images.squeeze()
        else:
            images = None

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels] 
        positives_mask = torch.from_numpy(np.array(positives_mask))
        negatives_mask = torch.from_numpy(np.array(negatives_mask))              
        poses = torch.from_numpy(np.array(poses))

        if params.model_params.use_bev:
            batch = {'orig_pc': clouds, 'pc': bevs, 'img': images}
        elif params.model_params.use_range_image:
            batch = {'orig_pc': clouds, 'depth': depths, 'img': images}
        else:
            batch = {'orig_pc': clouds, 'pc': clouds_quant, 'img': images}

        # Returns (batch_size, n_points, 3) tensor and positives_mask and negatives_mask which are
        # batch_size x batch_size boolean tensors
        # return batch, positives_mask, negatives_mask, torch.tensor(sampled_positive_ndx), torch.tensor(relative_poses)

        return batch, positives_mask, negatives_mask, poses, depth_maps

    return collate_fn


def make_dataloaders(params: TrainingParams, debug=False, device='cpu', validation=True):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, validation=validation)

    dataloders = {}
    train_sampler = BatchSampler(datasets['global_train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['global_train'], params)
    dataloders['global_train'] = DataLoader(datasets['global_train'], batch_sampler=train_sampler,
                                            collate_fn=train_collate_fn, num_workers=params.num_workers,
                                            pin_memory=True) # pin_memory=False
    if validation and 'global_val' in datasets:
        val_collate_fn = make_collate_fn(datasets['global_val'], params)
        val_sampler = BatchSampler(datasets['global_val'], batch_size=params.batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['global_val'] = DataLoader(datasets['global_val'], batch_sampler=val_sampler,
                                              collate_fn=val_collate_fn,
                                              num_workers=params.num_workers, pin_memory=True) # pin_memory=False

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] radius")
    return filtered_query_set


def preprocess_pointcloud(pc, remove_zero_points: bool = False,
                 min_x: float = None, max_x: float = None,
                 min_y: float = None, max_y: float = None,
                 min_z: float = None, max_z: float = None):
    if remove_zero_points:
        mask = np.all(np.isclose(pc, 0.), axis=1)
        pc = pc[~mask]

    if min_x is not None:
        mask = pc[:, 0] > min_x
        pc = pc[mask]

    if max_x is not None:
        mask = pc[:, 0] <= max_x
        pc = pc[mask]

    if min_y is not None:
        mask = pc[:, 1] > min_y
        pc = pc[mask]

    if max_y is not None:
        mask = pc[:, 1] <= max_y
        pc = pc[mask]

    if min_z is not None:
        mask = pc[:, 2] > min_z
        pc = pc[mask]

    if max_z is not None:
        mask = pc[:, 2] <= max_z
        pc = pc[mask]

    return pc


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e


def rotation_on_depth(depth, rotation):
    # rotation to translation [-180:180] -> [-cfg.num_sector//2:cfg.num_sector//2]
    if rotation > 0:
        t = rotation / 180. * (900 // 2)
        t = np.floor(t).astype(int)
        patch = depth[:, (900-t):900]
        col, row = 900, 64
        center = (col // 2, row // 2)
        t_x, t_y = t, 0.

        M = cv2.getRotationMatrix2D(center, 0.0, 1.0)
        depth = cv2.warpAffine(depth, M, (col, row))

        N = np.float32([[1,0,t_x],[0,1,t_y]])
        depth = cv2.warpAffine(depth, N, (col, row))
        depth[:, 0:t] = patch
    else:
        t = -rotation / 180. * (900 // 2)
        t = np.floor(t).astype(int)
        patch = depth[:, 0:t]
        col, row = 900, 64
        center = (col // 2, row // 2)
        t_x, t_y = -t, 0.

        M = cv2.getRotationMatrix2D(center, 0.0, 1.0)
        depth = cv2.warpAffine(depth, M, (col, row))

        N = np.float32([[1,0,t_x],[0,1,t_y]])
        depth = cv2.warpAffine(depth, N, (col, row))
        depth[:, (900-t):900] = patch
    # plt.figure()
    # plt.imshow(depth)
    # plt.show()
    # plt.savefig('./depth_rot.png')
    return depth