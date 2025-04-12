# Training tuples generation for NCLT dataset.

import numpy as np
import argparse
import tqdm
import pickle
import os
import cv2
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from glnet.config.config import *
from glnet.datasets.nclt.nclt_raw import NCLTSequence, NCLTSequences, load_lidar_file_nclt, load_im_file_for_generate, pc2image_file, calculate_T
from glnet.datasets.base_datasets import TrainingTuple
from glnet.datasets.nclt.utils import relative_pose
from glnet.utils.common_utils import _ex
from glnet.utils.data_utils.point_clouds import visualize_2d_data, generate_bev, generate_bev_occ, icp, o3d_icp
from glnet.datasets.panorama import generate_sph_image
from glnet.utils.data_utils.poses import m2ypr

DEBUG = False
ICP_REFINE = False

bounds = (nclt_pc_bev_conf['x_bound'][0], nclt_pc_bev_conf['x_bound'][1], nclt_pc_bev_conf['y_bound'][0], \
          nclt_pc_bev_conf['y_bound'][1], nclt_pc_bev_conf['z_bound'][0], nclt_pc_bev_conf['z_bound'][1])

def generate_training_tuples(ds: NCLTSequences, pos_threshold: float = 25, neg_threshold: float = 50, bev: bool = False, sph: bool = False):
    # displacement: displacement between consecutive anchors (if None all scans are takes as anchors).
    #               Use some small displacement to ensure there's only one scan if the vehicle does not move

    tuples = {}   # Dictionary of training tuples: tuples[ndx] = (sef ot positives, set of non negatives)
    for anchor_ndx in tqdm.tqdm(range(len(ds))):
        if bev:
            reading_filepath = os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx])
            bev_filename = reading_filepath.replace('bin', 'npy')
            bev_filename = bev_filename.replace('velodyne_sync', 'bev')
            if os.path.exists(bev_filename):
                pass
            else: 
                pc = load_lidar_file_nclt(reading_filepath).astype(np.float32)
                pc_bev = generate_bev(pc, bounds=bounds).numpy()
                print(f'Generating {bev_filename}')
                np.save(bev_filename, pc_bev)
                # for i in range(20):
                #     visualize_2d_data(pc_bev[i], f'pc_bev_{i}.jpg')
                #     visualize_2d_data(pc_bev_2[i], f'pc_bev_2_{i}.jpg')
                                         
        if sph:
            reading_filepath = os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx])
            sph_filename = reading_filepath.replace('bin', 'jpg')
            sph_filename = sph_filename.replace('velodyne_sync', 'sph')
            # if os.path.exists(sph_filename):
            #     pass
            # else:
            images = [load_im_file_for_generate(pc2image_file(reading_filepath, 'velodyne_sync/', i, '.bin'), False) for i in range(1, 6)]                
            sph_img = generate_sph_image(images, 'nclt', ds.dataset_root)
            print(f'Generating {sph_filename}')
            cv2.imwrite(sph_filename, sph_img)
            query_yaw, pitch, roll = m2ypr(ds.poses[anchor_ndx])

        anchor_pos = ds.get_xy()[anchor_ndx]

        # Find timestamps of positive and negative elements
        positives = ds.find_neighbours_ndx(anchor_pos, pos_threshold)
        non_negatives = ds.find_neighbours_ndx(anchor_pos, neg_threshold)
        # Remove anchor element from positives, but leave it in non_negatives
        positives = positives[positives != anchor_ndx]

        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)
        
        # ICP pose refinement
        fitness_l = []
        inlier_rmse_l = []
        positive_poses = {}
        
        if DEBUG:
            # Use ground truth transform without pose refinement
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pose = ds.poses[positive_ndx]
                # Compute initial relative pose
                m, fitness, inlier_rmse = relative_pose(anchor_pose, positive_pose), 1., 1.
                fitness_l.append(fitness)
                inlier_rmse_l.append(inlier_rmse)
                positive_poses[positive_ndx] = m
        else:
            anchor_pc = load_lidar_file_nclt(os.path.join(ds.dataset_root, ds.rel_scan_filepath[anchor_ndx])).astype(np.float32)
            anchor_pose = ds.poses[anchor_ndx]
            for positive_ndx in positives:
                positive_pose = ds.poses[positive_ndx]
                transform = relative_pose(anchor_pose, positive_pose)
                if ICP_REFINE:
                    positive_pc = load_lidar_file_nclt(os.path.join(ds.dataset_root, ds.rel_scan_filepath[positive_ndx])).astype(np.float32)
                    # Compute initial relative pose
                    # Refine the pose using ICP
                    m, fitness, inlier_rmse = icp(anchor_pc[:, :3], positive_pc[:, :3], transform)

                    fitness_l.append(fitness)
                    inlier_rmse_l.append(inlier_rmse)
                    positive_poses[positive_ndx] = m
                positive_poses[positive_ndx] = transform

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        tuples[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=ds.timestamps[anchor_ndx],
                                           rel_scan_filepath=ds.rel_scan_filepath[anchor_ndx],
                                           positives=positives, non_negatives=non_negatives, pose=anchor_pose,
                                           positives_poses=positive_poses)

    print(f'{len(tuples)} training tuples generated')
    if ICP_REFINE:
        print('ICP pose refimenement stats:')
        print(f'Fitness - min: {np.min(fitness_l):0.3f}   mean: {np.mean(fitness_l):0.3f}   max: {np.max(fitness_l):0.3f}')
        print(f'Inlier RMSE - min: {np.min(inlier_rmse_l):0.3f}   mean: {np.mean(inlier_rmse_l):0.3f}   max: {np.max(inlier_rmse_l):0.3f}')

    return tuples


def generate_image_meta_pickle(dataset_root: str):
    cam_params_path = dataset_root + '/cam_params/'
    K = []
    T = []    
    factor_x = 224. / 600.
    factor_y = 384. / 900.    
    for cam_num in range(1, 6):
        K_matrix = np.loadtxt(cam_params_path + 'K_cam%d.csv' % (cam_num), delimiter=',')
        fx = K_matrix[0][0]
        fy = K_matrix[1][1]
        cx = K_matrix[0][2]
        cy = K_matrix[1][2]

        cy = 1232. - cy 
        cx -= 400.  # cx
        cy -= 182.  # cy
        cx = cx * factor_x
        cy = cy * factor_y
        
        K_matrix[0][0] = fy * factor_y
        K_matrix[0][2] = cy
        K_matrix[1][1] = fx * factor_x
        K_matrix[1][2] = cx
        K.append(K_matrix)
        
        T_matrix = np.loadtxt(cam_params_path + 'x_lb3_c%d.csv' % (cam_num), delimiter=',')
        T_matrix = calculate_T(T_matrix)
        T.append(T_matrix)
    
    image_meta = {'K': K, 'T': T}
    with open(os.path.join(dataset_root, 'image_meta.pkl'), 'wb') as handle:
        pickle.dump(image_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training tuples')
    parser.add_argument('--dataset_root', default='~/Data/NCLT')
    parser.add_argument('--pos_threshold', default=25.0)
    parser.add_argument('--neg_threshold', default=50.0)
    parser.add_argument('--sampling_distance', type=float, default=0.2)
    parser.add_argument('--bev', action='store_true', help='Generate bevs projected by point clouds')
    parser.add_argument('--sph', action='store_true', help='Generate panorama images')
    args = parser.parse_args()
    
    sequences = ['2012-01-08']
    # sequences = ['2012-02-04', '2012-03-17']
    # sequences = ['2012-02-04', '2012-03-17', '2012-05-26', '2013-04-05']
    
    dataset_root = _ex(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Sequences: {sequences}')
    print(f'Threshold for positive examples: {args.pos_threshold}')
    print(f'Threshold for negative examples: {args.neg_threshold}')
    print(f'Minimum displacement between consecutive anchors: {args.sampling_distance}')
    
    # generate image meta pickle
    if not os.path.exists(os.path.join(dataset_root, 'image_meta.pkl')):
        generate_image_meta_pickle(dataset_root)
    
    split = 'all' # 'train' or 'all'
    ds = NCLTSequences(dataset_root, sequences, split=split, sampling_distance=args.sampling_distance)
    train_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold, bev=args.bev, sph=args.sph)
    name = ''
    for seq in sequences:
        name += '_' + seq
    pickle_name = f'{split}{name}_{args.pos_threshold}_{args.neg_threshold}_{args.sampling_distance}.pickle'
    train_tuples_filepath = os.path.join(dataset_root, pickle_name)
    pickle.dump(train_tuples, open(train_tuples_filepath, 'wb'))
    train_tuples = None
    
    split = 'test'
    ds = NCLTSequences(dataset_root, sequences, split=split, sampling_distance=args.sampling_distance)
    print('test sequences len: ', len(ds))
    test_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold, bev=args.bev, sph=args.sph)
    print('test tuple length: ', len(test_tuples))
    pickle_name = f'val{name}_{args.pos_threshold}_{args.neg_threshold}_{args.sampling_distance}.pickle'
    test_tuples_filepath = os.path.join(dataset_root, pickle_name)
    pickle.dump(test_tuples, open(test_tuples_filepath, 'wb'))
    