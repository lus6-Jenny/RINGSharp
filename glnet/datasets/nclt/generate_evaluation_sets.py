# Test sets for NCLT dataset.

import argparse
from typing import List
import os
import copy
import tqdm
import numpy as np
from glnet.config.config import *
from glnet.datasets.nclt.nclt_raw import NCLTSequence, load_lidar_file_nclt, load_im_file_for_generate, pc2image_file
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet
from glnet.datasets.dataset_utils import filter_query_elements
from glnet.datasets.panorama import generate_sph_image
from glnet.utils.data_utils.point_clouds import visualize_2d_data, generate_bev
import cv2

DEBUG = False

bounds = (nclt_pc_bev_conf['x_bound'][0], nclt_pc_bev_conf['x_bound'][1], nclt_pc_bev_conf['y_bound'][0], \
          nclt_pc_bev_conf['y_bound'][1], nclt_pc_bev_conf['z_bound'][0], nclt_pc_bev_conf['z_bound'][1])

def get_scans(sequence: NCLTSequence) -> List[EvaluationTuple]:
    # Get a list of all readings from the test area in the sequence
    elems = []
    for ndx in range(len(sequence)):
        pose = sequence.poses[ndx]
        position = pose[:2, 3]
        item = EvaluationTuple(sequence.timestamps[ndx], sequence.rel_scan_filepath[ndx], position=position, pose=pose)
        elems.append(item)
    return elems


def generate_evaluation_set(dataset_root: str, map_sequence: str, query_sequence: str, split: str = 'test', bev: bool = False, sph: bool = False,
                            map_sampling_distance: float = 0.2, query_sampling_distance: float = 0.2, dist_threshold = 25) -> EvaluationSet:
    query_bev_folder = os.path.join(dataset_root, query_sequence, 'bev')
    map_bev_folder = os.path.join(dataset_root, map_sequence, 'bev')      
    query_sph_folder = os.path.join(dataset_root, query_sequence, 'sph')
    map_sph_folder = os.path.join(dataset_root, map_sequence, 'sph')     
    map_sequence = NCLTSequence(dataset_root, map_sequence, split=split, sampling_distance=map_sampling_distance)
    query_sequence = NCLTSequence(dataset_root, query_sequence, split=split, sampling_distance=query_sampling_distance)

    map_set = get_scans(map_sequence)
    query_set = get_scans(query_sequence)
    
    if bev:
        os.makedirs(query_bev_folder, exist_ok=True)
        os.makedirs(map_bev_folder, exist_ok=True)
        for i in tqdm.tqdm(range(len(query_set))):
            reading_filepath = query_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            bev_filename = reading_filepath.replace('bin', 'npy')
            bev_filename = bev_filename.replace('velodyne_sync', 'bev')            
            # if os.path.exists(bev_filename):
            #     pass
            # else:
            pc = load_lidar_file_nclt(reading_filepath).astype(np.float32)
            pc_bev = generate_bev(pc, bounds=bounds).numpy()
            print(f'Generating {bev_filename}')
            np.save(bev_filename, pc_bev)

        for i in tqdm.tqdm(range(len(map_set))):
            reading_filepath = map_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            bev_filename = reading_filepath.replace('bin', 'npy')
            bev_filename = bev_filename.replace('velodyne_sync', 'bev')            
            # if os.path.exists(bev_filename):
            #     pass
            # else:            
            pc = load_lidar_file_nclt(reading_filepath).astype(np.float32)
            pc_bev = generate_bev(pc, bounds=bounds).numpy()
            print(f'Generating {bev_filename}')
            np.save(bev_filename, pc_bev)
                                
    if sph:
        os.makedirs(query_sph_folder, exist_ok=True)
        os.makedirs(map_sph_folder, exist_ok=True)            
        for i in tqdm.tqdm(range(len(query_set))):
            reading_filepath = query_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            sph_filename = reading_filepath.replace('bin', 'jpg')
            sph_filename = sph_filename.replace('velodyne_sync', 'sph')            
            # if os.path.exists(sph_filename):
            #     pass
            # else:            
            images = [load_im_file_for_generate(pc2image_file(reading_filepath, '/velodyne_sync/', i, '.bin'), False) for i in range(1, 6)]
            sph_img = generate_sph_image(images, 'nclt', dataset_root)
            print(f'Generating {sph_filename}')
            cv2.imwrite(sph_filename, sph_img)

        for i in tqdm.tqdm(range(len(map_set))):
            reading_filepath = map_set[i].rel_scan_filepath
            reading_filepath = os.path.join(dataset_root, reading_filepath)
            sph_filename = reading_filepath.replace('bin', 'jpg')
            sph_filename = sph_filename.replace('velodyne_sync', 'sph')
            # if os.path.exists(sph_filename):
            #     pass
            # else:            
            images = [load_im_file_for_generate(pc2image_file(reading_filepath, '/velodyne_sync/', i, '.bin'), False) for i in range(1, 6)]
            sph_img = generate_sph_image(images, 'nclt', dataset_root)
            print(f'Generating {sph_filename}')
            cv2.imwrite(sph_filename, sph_img)

    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    query_set = filter_query_elements(query_set, map_set, dist_threshold)

    print(f'{len(map_set)} database elements, {len(query_set)} query elements')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for NCLT dataset')
    parser.add_argument('--dataset_root', type=str, default='~/Data/NCLT')
    parser.add_argument('--map_sampling_distance', type=float, default=20.0) # 20.0, 30.0, ..., 100.0
    parser.add_argument('--query_sampling_distance', type=float, default=5.0)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=50) # 25
    parser.add_argument('--bev', action='store_true', help='Generate bevs projected by point clouds')
    parser.add_argument('--sph', action='store_true', help='Generate panorama images')
    args = parser.parse_args()
    
    dataset_root = os.path.expanduser(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Map minimum displacement between consecutive anchors: {args.map_sampling_distance}')
    print(f'Query minimum displacement between consecutive anchors: {args.query_sampling_distance}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    # Sequences is a list of (map sequence, query sequence)
    multi_sequences = [
                    #    ('2012-02-04', '2012-03-17'),
                       ('2012-01-08', '2012-08-20'),
                    #    ('2012-01-08', '2012-11-16'),
                    #    ('2012-01-08', '2012-12-01'),
                    #    ('2012-08-20', '2012-11-16'),
                    #    ('2012-08-20', '2012-12-01'),
                    #    ('2012-11-16', '2012-12-01')
                       ]
    
    split = 'all' # 'test' or 'all'
    for sequences in multi_sequences:
        map_sequence, query_sequence = sequences
        print(f'Map sequence: {map_sequence}')
        print(f'Query sequence: {query_sequence}')

        test_set = generate_evaluation_set(dataset_root, map_sequence, query_sequence, split=split, bev=args.bev, sph=args.sph,
                map_sampling_distance=args.map_sampling_distance, query_sampling_distance=args.query_sampling_distance, dist_threshold=args.dist_threshold)

        pickle_name = f'{split}_{map_sequence}_{query_sequence}_{args.map_sampling_distance}_{args.query_sampling_distance}.pickle'
        file_path_name = os.path.join(dataset_root, pickle_name)
        test_set.save(file_path_name)
