# Training tuples generation for Oxford dataset.

import numpy as np
import argparse
import tqdm
import pickle
import os
import cv2
from glnet.datasets.oxford.oxford_raw import OxfordSequence, OxfordSequences, OxfordPointCloudLoader, load_img_file_oxford
from glnet.datasets.base_datasets import TrainingTuple
from glnet.datasets.oxford.utils import *
from glnet.config.config import *
from glnet.utils.common_utils import _ex
from glnet.utils.data_utils.point_clouds import visualize_2d_data, generate_bev, icp, o3d_icp, fast_gicp
from glnet.datasets.panorama import generate_sph_image
import tqdm
import time

ICP_REFINE = False
bounds = (oxford_pc_bev_conf['x_bound'][0], oxford_pc_bev_conf['x_bound'][1], oxford_pc_bev_conf['y_bound'][0], \
          oxford_pc_bev_conf['y_bound'][1], oxford_pc_bev_conf['z_bound'][0], oxford_pc_bev_conf['z_bound'][1])

def get_model_name(model):
    if model == 'stereo':
        return 'stereo_narrow_left'
    else:
        return model


def generate_image_meta_pickle(dataset_root: str):
    models = ['mono_left', 'mono_right', 'mono_rear', 'stereo']

    intrinsic_path = os.path.join(dataset_root, 'models')
    extrinsic_path = os.path.join(dataset_root, 'extrinsics')

    factor_x_mono = 320. / 512.
    factor_y_mono = 320. / 512.
    factor_x_stereo = 640. / 1280.
    factor_y_stereo = 320. / 640.

    K = []
    T = []
    for model in models:
        model_K = get_model_name(model)
        if model == 'mono_left':
            factor_x = factor_x_mono
            factor_y = factor_y_mono
            dy = 200
        else:
            factor_x = factor_x_stereo
            factor_y = factor_y_stereo
            dy = 160

        intrinsics_path = os.path.join(intrinsic_path, model_K + '.txt')
        K_matrix = np.eye(3)

        with open(intrinsics_path) as intrinsics_file:
            vals = [float(x) for x in next(intrinsics_file).split()]
            K_matrix[0][0] = vals[0] * factor_x
            K_matrix[1][1] = vals[1] * factor_y
            K_matrix[0][2] = vals[2] * factor_x
            K_matrix[1][2] = (vals[3] - dy) * factor_y

            T_cam_matrix = []
            for line in intrinsics_file:
                T_cam_matrix.append([float(x) for x in line.split()])
            T_cam_matrix = np.array(T_cam_matrix)

        K.append(K_matrix)

        extrinsics_path = os.path.join(extrinsic_path, model + '.txt')
        with open(extrinsics_path) as extrinsics_file:
            extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
            T_matrix = build_se3_transform(extrinsics)
            T.append(np.dot(np.linalg.inv(T_cam_matrix), T_matrix))

    image_meta = {'K': K, 'T': T}
    with open(dataset_root, 'image_meta.pkl', 'wb') as handle:
        pickle.dump(image_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_training_tuples(ds: OxfordSequences, pos_threshold: float = 25, neg_threshold: float = 50, bev: bool = False, sph: bool = False):
    # displacement: displacement between consecutive anchors (if None all scans are takes as anchors).
    #               Use some small displacement to ensure there's only one scan if the vehicle does not move

    tuples = {}   # Dictionary of training tuples: tuples[ndx] = (sef ot positives, set of non negatives)
    pc_loader = OxfordPointCloudLoader()        
    for anchor_ndx in tqdm.tqdm(range(len(ds))):
        file_pathname = ds.filepaths[anchor_ndx]
        pc, _ = pc_loader(file_pathname)
        if bev:            
            bev_filename = file_pathname[2].replace('png', 'npy')
            bev_filename = bev_filename.replace('mono_left_rect', 'bev')
            # if os.path.exists(bev_filename):
            #     pass
            # else:
            pc_bev = generate_bev(pc, bounds=bounds).numpy()
            # for i in range(pc_bev.shape[0]):
            #     visualize_2d_data(pc_bev[i], f'bev_{i}.jpg')         
            print(f'Generating {bev_filename}')
            np.save(bev_filename, pc_bev)
        if sph:
            sph_filename = file_pathname[2].replace('mono_left_rect', 'sph')  
            # if os.path.exists(sph_filename):
            #     pass
            # else:                           
            images = [load_img_file_oxford(file_pathname[i]) for i in range(2, 6)]
            sph_img = generate_sph_image(images, 'oxford', ds.dataset_root)
            print(f'Generating {sph_filename}')
            cv2.imwrite(sph_filename, sph_img)
        
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

        # Use ground truth transform without pose refinement
        anchor_pc, _ = pc_loader(file_pathname)
        anchor_pose = ds.poses[anchor_ndx]
        for positive_ndx in positives:
            positive_pose = ds.poses[positive_ndx]
            # Compute initial relative pose
            m, fitness, inlier_rmse = relative_pose(anchor_pose, positive_pose), 1., 1.
            if ICP_REFINE:
                positive_pc, _ = pc_loader(ds.filepaths[positive_ndx])
                # time_start = time.time()
                # Refine the pose using ICP
                # m, fitness, inlier_rmse = icp(anchor_pc[:,:3], positive_pc[:,:3], m)
                # _, m, _ = o3d_icp(anchor_pc[:,:3], positive_pc[:,:3], transform=m, point2plane=True)    
                _, m = fast_gicp(anchor_pc[:,:3], positive_pc[:,:3], init_pose=m) # ! spend the smallest time
                # time_end = time.time()       
                # print(f'Time of ICP: {time_end-time_start:.4f}')
            fitness_l.append(fitness)
            inlier_rmse_l.append(inlier_rmse)
            positive_poses[positive_ndx] = m

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        tuples[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=ds.timestamps[anchor_ndx],
                                           rel_scan_filepath=ds.rel_scan_filepath[anchor_ndx],
                                           positives=positives, non_negatives=non_negatives, pose=anchor_pose,
                                           positives_poses=positive_poses, filepaths=ds.filepaths[anchor_ndx])

    print(f'{len(tuples)} training tuples generated')

    return tuples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training tuples')
    parser.add_argument('--dataset_root', type=str, default='~/Data/Oxford_radar')
    parser.add_argument('--pos_threshold', default=25.0)
    parser.add_argument('--neg_threshold', default=50.0)
    parser.add_argument('--sampling_distance', type=float, default=0.2)
    parser.add_argument('--bev', action='store_true', help='Generate bevs projected by point clouds')
    parser.add_argument('--sph', action='store_true', help='Generate panorama images')    
    args = parser.parse_args()
    
    dataset_root = _ex(args.dataset_root)
    sequences = ['2019-01-11-13-24-51', '2019-01-15-13-06-37']

    print(f'Dataset root: {dataset_root}')
    print(f'Sequences: {sequences}')
    print(f'Threshold for positive examples: {args.pos_threshold}')
    print(f'Threshold for negative examples: {args.neg_threshold}')
    print(f'Minimum displacement between consecutive anchors: {args.sampling_distance}')
    
    # generate image meta pickle
    if not os.path.exists(os.path.join(dataset_root, 'image_meta.pkl')):
        generate_image_meta_pickle(dataset_root)
        
    split = 'all' # 'train' or 'all'
    ds = OxfordSequences(dataset_root, sequences, split=split, sampling_distance=args.sampling_distance)
    train_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold, bev=args.bev, sph=args.sph)
    name = ''
    for seq in sequences:
        name += '_' + seq    
    pickle_name = f'{split}{name}_{args.pos_threshold}_{args.neg_threshold}_{args.sampling_distance}.pickle'
    train_tuples_filepath = os.path.join(dataset_root, pickle_name)
    pickle.dump(train_tuples, open(train_tuples_filepath, 'wb'))
    train_tuples = None
    
    split = 'test'
    ds = OxfordSequences(dataset_root, sequences, split=split, sampling_distance=args.sampling_distance)
    print('test sequences len: ', len(ds))
    test_tuples = generate_training_tuples(ds, args.pos_threshold, args.neg_threshold, bev=args.bev, sph=args.sph)
    print('test tuple length: ', len(test_tuples))
    pickle_name = f'val{name}_{args.pos_threshold}_{args.neg_threshold}_{args.sampling_distance}.pickle'
    test_tuples_filepath = os.path.join(dataset_root, pickle_name)
    pickle.dump(test_tuples, open(test_tuples_filepath, 'wb'))
