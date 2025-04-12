# Functions and classes operating on a raw NCLT dataset

import numpy as np
import cv2
import os
from typing import List
from torch.utils.data import Dataset, ConcatDataset
from sklearn.neighbors import KDTree
import torch
import random
import struct
from glnet.datasets.nclt.utils import read_lidar_poses, in_test_split, in_train_split
from glnet.utils.data_utils.point_clouds import PointCloudLoader, PointCloudWithImageLoader
from glnet.utils.common_utils import ssc_to_homo
import matplotlib.pyplot as plt


# nclt pointcloud utils
def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset
    
    return x, y, z
    

# load lidar file in nclt dataset
def load_lidar_file_nclt(file_path):
    f_bin = open(file_path,'rb')
    hits = []
    while True:
        x_str = f_bin.read(2)
        if x_str == b"": # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]
        x, y, z = convert(x, y, z)
        hits += [[x, y, z]]

    f_bin.close()
    hits = np.asarray(hits)

    return hits


def calculate_T(x_lb_c):
    '''
    Ground Truth Pose is in Body/Velodyne Frame
    '''
    # T_y_x: x in y coordinate
    x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]
    x_body_lidar = [0.002, -0.004, -0.957, 0.807, 0.166, -90.703]
    x_camNormal_cam = [0.0, 0.0, 0.0, 0.0, 0.0, 90.0]

    T_lb3_c = ssc_to_homo(x_lb_c)
    T_body_lb3 = ssc_to_homo(x_body_lb3)
    T_body_lidar = ssc_to_homo(x_body_lidar)
    T_camNormal_cam = ssc_to_homo(x_camNormal_cam)
    
    T_lb3_body = np.linalg.inv(T_body_lb3)
    T_c_lb3 = np.linalg.inv(T_lb3_c)

    T_c_body = np.matmul(T_c_lb3, T_lb3_body)
    T_c_lidar = np.matmul(T_c_body, T_body_lidar)
    T_camNormal_body = np.matmul(T_camNormal_cam, T_c_body)
    T_camNormal_lidar = np.matmul(T_camNormal_cam, T_c_lidar)
    
    return T_camNormal_body


def pc2image_file(pc_filename, vel_folder, cam_num, vel_type):
    img_filename = pc_filename.replace(vel_type, '.jpg')
    img_filename = img_filename.replace(vel_folder, '/lb3_u_s_384/Cam' + str(cam_num) + "/")
    # print(img_filename)
    return img_filename


def load_im_file_for_generate(filename, sph):
    assert os.path.exists(filename), f'Cannot access image file: {filename}'
    input_image = cv2.imread(filename)
    input_image = cv2.cvtColor(input_image, cv2.IMREAD_COLOR) # we trained our model and BEV-NetVLAD-MLP with BGR image as inputs
    # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) # compared methods used RGB image as inputs

    if sph:
        pass
    else:
        input_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE) # ROTATE_90_COUNTERCLOCKWISE

    return input_image


class NCLTPointCloudWithImageLoader(PointCloudWithImageLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -0.5

    def read_pcim(self, file_pathname: str, sph=False, extrinsics_dir=None) -> torch.Tensor:
        # Reads the point cloud without pre-processing
        # Returns Nx4 tensor
        # pc = np.fromfile(file_pathname, dtype=np.float32)
        assert os.path.exists(file_pathname), f'Cannot access point cloud file: {file_pathname}'
        if file_pathname.split('.')[-1] == 'npy':
            pc = np.load(file_pathname)
            mask = pc[:, 2] < self.ground_plane_level
            pc = pc[mask]
            if sph:
                file_pathname = file_pathname.replace('.npy', '.jpg')
                file_pathname = file_pathname.replace('submap', 'sph')
                images = [load_im_file_for_generate(file_pathname, sph)]
            else:
                images = [load_im_file_for_generate(pc2image_file(file_pathname, '/submap/', i, '.npy'), sph) for i in range(1, 6)]            
        else:
            pc = load_lidar_file_nclt(file_pathname)
            mask = pc[:, 2] < self.ground_plane_level
            pc = pc[mask]
            if sph:
                file_pathname = file_pathname.replace('.bin', '.jpg')
                file_pathname = file_pathname.replace('velodyne_sync', 'sph')
                images = [load_im_file_for_generate(file_pathname, sph)]
            else:
                images = [load_im_file_for_generate(pc2image_file(file_pathname, '/velodyne_sync/', i, '.bin'), sph) for i in range(1, 6)]

        return pc, images


class NCLTPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level.
        self.ground_plane_level = -0.5

    def read_pc(self, file_pathname: str, sph=False, extrinsics_dir=None) -> torch.Tensor:
        # Reads the point cloud without pre-processing
        # Returns Nx3 tensor
        pc = load_lidar_file_nclt(file_pathname)
        mask = pc[:, 2] < self.ground_plane_level
        pc = pc[mask]
        img = None
        return pc, img


class NCLTSequence(Dataset):
    """
    Dataset returns a point cloud from a train or test split from one sequence from a raw NCLT dataset
    """
    def __init__(self, dataset_root: str, sequence_name: str, split: str, sampling_distance: float = 0.2):
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_name = sequence_name
        sequence_path = os.path.join(self.dataset_root, self.sequence_name)
        assert os.path.exists(sequence_path), f'Cannot access sequence: {sequence_path}'
        self.split = split
        self.sampling_distance = sampling_distance
        # Maximum discrepancy between timestamps of LiDAR scan and global pose in seconds
        self.pose_time_tolerance = 1.

        self.pose_file = os.path.join(sequence_path, 'groundtruth_'+self.sequence_name+'.csv')
        assert os.path.exists(self.pose_file), f'Cannot access ground truth file: {self.pose_file}'

        self.rel_lidar_path = os.path.join(self.sequence_name, 'velodyne_sync')
        lidar_path = os.path.join(self.dataset_root, self.rel_lidar_path)
        assert os.path.exists(lidar_path), f'Cannot access lidar scans: {lidar_path}'
        image_path = os.path.join(self.dataset_root, self.sequence_name, 'lb3_u_s_384/Cam1')
        assert os.path.exists(image_path), f'Cannot access images: {image_path}'
        self.pcim_loader = NCLTPointCloudWithImageLoader()

        timestamps, poses = read_lidar_poses(self.pose_file, lidar_path, image_path, self.pose_time_tolerance)
        self.xys = poses[:, :2, 3]
        self.timestamps, self.poses = self.filter(timestamps, poses)
        self.rel_scan_filepath = [os.path.join(self.rel_lidar_path, str(e) + '.bin') for e in self.timestamps]

        assert len(self.timestamps) == len(self.poses)
        assert len(self.timestamps) == len(self.rel_scan_filepath)
        print(f'{len(self.timestamps)} scans in {sequence_name}-{split}')

    def __len__(self):
        return len(self.rel_scan_filepath)

    def __getitem__(self, ndx):
        reading_filepath = os.path.join(self.dataset_root, self.rel_scan_filepath[ndx])
        pcs, images = self.pcim_loader(reading_filepath)

        return {'pc': pcs, 'img': images, 'pose': self.poses[ndx], 'ts': self.timestamps[ndx],
                'position': self.poses[ndx][:2, 3]}

    def load_pcs(self, scan_paths):
        # Load point cloud from file
        pcs = []
        for scan_path in scan_paths:
            pc = load_lidar_file_nclt(scan_path)
            if len(pc) == 0:
                continue
            pcs.append(pc)
        pcs = np.array(pcs)
        return pcs 
        
    def filter(self, ts: np.ndarray, poses: np.ndarray):
        # Filter out scans - retain only scans within a given split with minimum displacement
        positions = poses[:, :2, 3]

        # Retain elements in the given split
        # Only sejong sequence has train/test split
        if self.split != 'all':
            if self.split == 'train':
                mask = in_train_split(positions)
            elif self.split == 'test':
                mask = in_test_split(positions)

            ts = ts[mask]
            poses = poses[mask]
            positions = positions[mask]
            print(f'Split: {self.split}   Mask len: {len(mask)}   Mask True: {np.sum(mask)}')

        # Filter out scans - retain only scans within a given split
        prev_position = None
        mask = []
        for ndx, position in enumerate(positions):
            if np.any(np.isnan(position)):
                continue
            if prev_position is None:
                mask.append(ndx)
                prev_position = position
            else:
                displacement = np.linalg.norm(prev_position - position)
                if displacement > self.sampling_distance:
                    mask.append(ndx)
                    prev_position = position

        ts = ts[mask]
        poses = poses[mask]
        return ts, poses


class NCLTSequences(Dataset):
    """
    Multiple NCLT sequences indexed as a single dataset. Each element is identified by a unique global index.
    """
    def __init__(self, dataset_root: str, sequence_names: List[str], split: str, sampling_distance: float = 0.5):
        assert len(sequence_names) > 0
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test', 'all']

        self.dataset_root = dataset_root
        self.sequence_names = sequence_names
        self.split = split
        self.sampling_distance = sampling_distance

        sequences = []
        for seq_name in self.sequence_names:
            ds = NCLTSequence(self.dataset_root, seq_name, split=split, sampling_distance=sampling_distance)
            sequences.append(ds)

        self.dataset = ConcatDataset(sequences)

        # Concatenate positions from all sequences
        self.poses = np.zeros((len(self.dataset), 4, 4), dtype=np.float64)
        self.timestamps = np.zeros((len(self.dataset),), dtype=np.int64)
        self.rel_scan_filepath = []
        self.xys = self.poses[:, :2, 3]
        
        for cum_size, ds in zip(self.dataset.cumulative_sizes, sequences):
            # Consolidated lidar positions, timestamps and relative filepaths
            self.poses[cum_size - len(ds): cum_size, :] = ds.poses
            self.timestamps[cum_size - len(ds): cum_size] = ds.timestamps
            self.rel_scan_filepath.extend(ds.rel_scan_filepath)

        assert len(self.timestamps) == len(self.poses)
        assert len(self.timestamps) == len(self.rel_scan_filepath)

        # Build a kdtree based on X, Y position
        self.kdtree = KDTree(self.get_xy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]

    def get_xy(self):
        # Get X, Y position from (4, 4) pose
        return self.poses[:, :2, 3]

    def load_pcs(self, scan_paths):
        # Load point cloud from file
        pcs = []
        for scan_path in scan_paths:
            pc = load_lidar_file_nclt(scan_path)
            if len(pc) == 0:
                continue
            pcs.append(pc)
        pcs = np.array(pcs)
        return pcs 

    def find_neighbours_ndx(self, position, radius):
        # Returns indices of neighbourhood point clouds for a given position
        assert position.ndim == 1
        assert position.shape[0] == 2
        # Reshape into (1, 2) axis
        position = position.reshape(1, -1)
        neighbours = self.kdtree.query_radius(position, radius)[0]
        random.shuffle(neighbours)

        return neighbours.astype(np.int32)


if __name__ == '__main__':
    dataset_root = os.path.expanduser('~/Data/NCLT/')
    sequence_names = ['2012-02-04']

    db = NCLTSequences(dataset_root, sequence_names, split='test')
    print(f'Number of scans in the sequence: {len(db)}')
    e = db[0]

    res = db.find_neighbours_ndx(e['position'], radius=50)
    print('.')

