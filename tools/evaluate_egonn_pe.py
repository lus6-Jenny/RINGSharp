# Warsaw University of Technology

import argparse

import numpy as np
import tqdm
import os
import random
from typing import List
import open3d as o3d
from time import time
import torch
import MinkowskiEngine as ME

from glnet.datasets.quantization import Quantizer
from glnet.utils.data_utils.point_clouds import icp, o3d_icp, fast_gicp, make_open3d_feature, make_open3d_point_cloud
from glnet.models.model_factory import model_factory
from glnet.utils.data_utils.poses import m2ypr, m2xyz_ypr, relative_pose, apply_transform
from glnet.utils.params import ModelParams
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from glnet.models.utils import *
from glnet.config.config import *

from tools.evaluator import Evaluator
from tools.plot_PR_curve import compute_PR_pairs
from tools.plot_pose_errors import plot_cdf, cal_recall_pe


class MinkLocGLEvaluator(Evaluator):
    # Evaluation of Global Localization
    default_config = {
        'positive_threshold': 10, # estimate relative pose of two images within 10 meters
        'rotation_threshold': 5, # rotation thershold to calculation pose estimation success rate
        'translation_threshold': 2, # translation thershold to calculation pose estimation success rate
    }
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float], k: int = 20, n_samples=None, repeat_dist_th: float = 0.5,
                 n_k: List[float] = (128, 256), quantizer: Quantizer = None,
                 icp_refine: bool = True, ignore_keypoint_saliency: bool = False, debug: bool = False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        assert quantizer is not None
        self.repeat_dist_th = repeat_dist_th
        self.quantizer = quantizer
        self.n_k = n_k
        self.icp_refine = icp_refine
        self.ignore_keypoint_saliency = ignore_keypoint_saliency

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self, model, exp_name=None, config=default_config, *args, **kwargs):
        if exp_name is None:
            exp_name = self.params.model
        if self.dataset_type == 'nclt':
            gt_icp_refine = False # use icp pose refinement
        elif self.dataset_type == 'oxford':
            gt_icp_refine = True # use icp pose refinement
        if 'only_global' in kwargs:
            self.only_global = kwargs['only_global']
        else:
            self.only_global = False
        
        map_embeddings, local_map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings, local_query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_xys = map_positions = self.eval_set.get_map_positions()
        tree = KDTree(map_positions)
        query_xys = query_positions = self.eval_set.get_query_positions()

        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')

        if self.only_global:
            metrics = {}
        else:
            metrics = {n_kpts: {'rre': [], 'rte': [], 'repeatability': [],
                                'success': [], 'success_inliers': [], 'failure_inliers': [],
                                'rre_refined': [], 'rte_refined': [], 'success_refined': [],
                                'success_inliers_refined': [], 'repeatability_refined': [],
                                'failure_inliers_refined': [], 't_ransac': []}
                       for n_kpts in self.n_k}

        total_positives = 0 # Total number of positives
        x_errors = []
        y_errors = []
        trans_errors = []
        yaw_errors = []
        real_dists = []
        positive_threshold = config['positive_threshold'] # Positive threshold for global localization
        rotation_threshold = config['rotation_threshold'] # Rotation threshold to calculation pose estimation success rate
        translation_threshold = config['translation_threshold'] # translation thershold to calculation pose estimation success rate        
        
        quantiles = [0.25, 0.5, 0.75, 0.95]
        eval_setting = self.eval_set_filepath.split('/')[-1].split('.pickle')[0]
        folder_path = os.path.expanduser(f'./results/{exp_name}/{eval_setting}/pe_{positive_threshold}')
        os.makedirs(folder_path, exist_ok=True)
        
        for query_ndx in tqdm.tqdm(range(num_queries)):
            # Check if the query element has a true match within each radius
            query_position = query_positions[query_ndx]
            query_pose = self.eval_set.query_set[query_ndx].pose
            if self.dataset_type == 'nclt':
                query_filepath = os.path.join(self.dataset_root, self.eval_set.query_set[query_ndx].rel_scan_filepath)
            elif self.dataset_type == 'oxford':
                query_filepath = self.eval_set.query_set[query_ndx].filepaths
            query_pc, _ = self.pcim_loader(query_filepath)
            
            nn_ndx = tree.query_radius(query_position.reshape(1,-1), positive_threshold)[0]
            num_positives = len(nn_ndx)
            if num_positives == 0: 
                continue
            total_positives += num_positives
                        
            # ------------ Pose Estimation ------------
            # Perform pose estimation for all positives within positive threshold
            for idx in nn_ndx:
                map_position = map_positions[idx]
                map_pose = self.eval_set.map_set[idx].pose
                if self.dataset_type == 'nclt':
                    map_filepath = os.path.join(self.dataset_root, self.eval_set.map_set[idx].rel_scan_filepath)
                elif self.dataset_type == 'oxford':
                    map_filepath = self.eval_set.map_set[idx].filepaths
                map_pc, _ = self.pcim_loader(map_filepath)
                
                T_gt = relative_pose(query_pose, map_pose)
                # Ground Truth Pose Refinement
                if gt_icp_refine:
                    # _, T_gt, _ = o3d_icp(query_pc[:,:3], map_pc[:,:3], transform=T_gt, point2plane=True)
                    _, T_gt = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=T_gt) # spends less time than open3d
                rel_x, rel_y, rel_z, rel_yaw, rel_pitch, rel_roll = m2xyz_ypr(T_gt)
                gt_d = np.sqrt(rel_x**2 + rel_y**2)
                
                for n_kpts in self.n_k:
                    # Get the first match and compute local stats only for the best match

                    # Ransac alignment
                    tick = time()
                    T_estimated, inliers, _ = self.ransac_fn(local_query_embeddings[query_ndx],
                                                            local_map_embeddings[idx], n_kpts)

                    t_ransac = time() - tick

                    # Refine the estimated pose using ICP
                    if self.icp_refine:
                        # T_estimated, _, _ = icp(query_pc[:,:3], map_pc[:,:3], T_estimated)
                        _, T_estimated = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=T_estimated)
                    
                    # Compute repeatability
                    kp1 = local_query_embeddings[query_ndx]['keypoints'][:n_kpts]
                    kp2 = local_map_embeddings[idx]['keypoints'][:n_kpts]
                    metrics[n_kpts]['repeatability'].append(calculate_repeatability(kp1, kp2, T_gt, threshold=self.repeat_dist_th))

                    # 3DoF errors
                    est_x, est_y, est_z, est_yaw, est_pitch, est_roll = m2xyz_ypr(T_estimated)
                    x_err = np.abs((rel_x - est_x)) # in meters
                    y_err = np.abs((rel_y - est_y)) # in meters
                    trans_err = np.sqrt(x_err**2 + y_err**2) # in meters
                    yaw_err = np.abs(angle_clip(rel_yaw - est_yaw)) * 180 / np.pi # in degrees
                    # 6DoF errors
                    rte = np.linalg.norm(T_estimated[:3, 3] - T_gt[:3, 3])
                    cos_rre = (np.trace(T_estimated[:3, :3].transpose(1, 0) @ T_gt[:3, :3]) - 1.) / 2.
                    rre = np.arccos(np.clip(cos_rre, a_min=-1., a_max=1.)) * 180. / np.pi
                    
                    metrics[n_kpts]['t_ransac'].append(t_ransac)    # RANSAC time

                    # 2 meters and 5 degrees threshold for successful registration
                    if rte > translation_threshold or rre > rotation_threshold:
                        metrics[n_kpts]['success'].append(0.)
                        metrics[n_kpts]['failure_inliers'].append(inliers)
                    else:
                        metrics[n_kpts]['success'].append(1.)
                        metrics[n_kpts]['rte'].append(rte)
                        metrics[n_kpts]['rre'].append(rre)
                        metrics[n_kpts]['success_inliers'].append(inliers)
            

                # Pose Error
                print(f'-------- Query {query_ndx+1} matched with map {idx+1} --------')
                print('Ground truth translation: x: {}, y: {}, rotation: {}'.format(rel_x, rel_y, rel_yaw))
                print('Estimated translation: x: {}, y: {}, rotation: {}'.format(est_x, est_y, est_yaw))
                
                x_errors.append(x_err)
                y_errors.append(y_err)
                trans_errors.append(trans_err)
                yaw_errors.append(yaw_err)    
                real_dists.append(gt_d)
        
        # Calculate mean metrics
        # ------------ Results ------------
        print(f'-------- Positive threshold: {positive_threshold} m --------')
        print(f'Number of positives: {total_positives}')
        mean_x_error = np.mean(x_errors)
        mean_y_error = np.mean(y_errors)
        mean_trans_error = np.mean(trans_errors)
        mean_yaw_error = np.mean(yaw_errors)
        x_error_quantiles = np.quantile(x_errors, quantiles)
        y_error_quantiles = np.quantile(y_errors, quantiles)
        trans_error_quantiles = np.quantile(trans_errors, quantiles)
        yaw_error_quantiles = np.quantile(yaw_errors, quantiles) 
        recall_pe = cal_recall_pe(yaw_errors, trans_errors, num_queries=total_positives, rot_thre=rotation_threshold, trans_thre=translation_threshold)               
        print(f'Mean x error at {positive_threshold} m: {mean_x_error}')    
        print(f'Mean y error at {positive_threshold} m: {mean_y_error}')
        print(f'Mean translation error at {positive_threshold} m: {mean_trans_error}')        
        print(f'Mean yaw error at {positive_threshold} m: {mean_yaw_error}')       
        print(f'X error quantiles at {positive_threshold} m: {x_error_quantiles}')
        print(f'Y error quantiles at {positive_threshold} m: {y_error_quantiles}')
        print(f'Translation error quantiles at {positive_threshold} m: {trans_error_quantiles}')
        print(f'Yaw error quantiles at {positive_threshold} m: {yaw_error_quantiles}')
        print(f'Recall at {translation_threshold} m and {rotation_threshold} degrees: {recall_pe}')
            
        np.save(f'{folder_path}/x_errors.npy', x_errors)
        np.save(f'{folder_path}/y_errors.npy', y_errors)
        np.save(f'{folder_path}/trans_errors.npy', trans_errors)
        np.save(f'{folder_path}/yaw_errors.npy', yaw_errors)
        np.save(f'{folder_path}/real_dists.npy', real_dists)
        
        global_metrics = {'dataset_type': self.dataset_type, 'eval_setting': eval_setting, 'positive_threshold': positive_threshold, 
                          'gt_icp_refine': gt_icp_refine, 'estimate_icp_refine': self.icp_refine, 'total_positives': total_positives, 'recall_pe': recall_pe, 'mean_x_error': mean_x_error, 
                          'mean_y_error': mean_y_error, 'mean_trans_error': mean_trans_error, 'mean_yaw_error': mean_yaw_error, 'x_error_quantiles': x_error_quantiles, 'y_error_quantiles': y_error_quantiles, 'trans_error_quantiles': trans_error_quantiles, 'yaw_error_quantiles': yaw_error_quantiles}        
        return global_metrics
    
    
    def ransac_fn(self, query_keypoints, candidate_keypoints, n_k):
        """

        Returns fitness score and estimated transforms
        Estimation using Open3d 6dof ransac based on feature matching.
        """
        kp1 = query_keypoints['keypoints'][:n_k]
        kp2 = candidate_keypoints['keypoints'][:n_k]
        ransac_result = get_ransac_result(query_keypoints['features'][:n_k], candidate_keypoints['features'][:n_k],
                                          kp1, kp2)
        return ransac_result.transformation, len(ransac_result.correspondence_set), ransac_result.fitness

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval((model,))
        global_embeddings = None
        local_embeddings = []
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            if self.dataset_type == 'nclt':
                scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                assert os.path.exists(scan_filepath)
            elif self.dataset_type == 'oxford':
                scan_filepath = e.filepaths
                assert os.path.exists(scan_filepath[0])
            pc, _ = self.pcim_loader(scan_filepath)
            pc = torch.tensor(pc, dtype=torch.float)

            global_embedding, keypoints, key_embeddings = self.compute_embedding(pc[:, :3], model)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[1]), dtype=global_embedding.dtype)

            global_embeddings[ndx] = global_embedding
            local_embeddings.append({'keypoints': keypoints, 'features': key_embeddings})

        return global_embeddings, local_embeddings

    def compute_embedding(self, pc, model, *args, **kwargs):
        """
        Returns global embedding (np.array) as well as keypoints and corresponding descriptors (torch.tensors)
        """
        coords, _ = self.quantizer(pc)
        with torch.no_grad():
            bcoords = ME.utils.batched_coordinates([coords])
            feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': bcoords.to(self.device), 'features': feats.to(self.device)}

            # Compute global descriptor
            y = model(batch)
            global_embedding = y['global'].detach().cpu().numpy()
            if 'descriptors' not in y:
                descriptors, keypoints, sigma, idxes = None, None, None, None
            else:
                descriptors, keypoints, sigma = y['descriptors'], y['keypoints'], y['sigma']
                # get sorted indexes of keypoints
                idxes = self.get_keypoints_idxes(sigma[0].squeeze(1), max(self.n_k))
                # keypoints and descriptors are in uncertainty increasing order
                keypoints = keypoints[0][idxes].detach().cpu()
                descriptors = descriptors[0][idxes].detach().cpu()

        return global_embedding, keypoints, descriptors

    def get_keypoints_idxes(self, sigmas, n_k):
        n_k = min(len(sigmas), n_k)
        if self.ignore_keypoint_saliency:
            # Get n_k random keypoints
            ndx = torch.randperm(len(sigmas))[:n_k]
        else:
            # Get n_k keypoints with the lowest uncertainty sorted in increasing order
            _, ndx = torch.topk(sigmas, dim=0, k=n_k, largest=False)

        return ndx


def get_ransac_result(feat1, feat2, kp1, kp2, ransac_dist_th=0.5, ransac_max_it=10000):
    feature_dim = feat1.shape[1]
    pcd_feat1 = make_open3d_feature(feat1, feature_dim, feat1.shape[0])
    pcd_feat2 = make_open3d_feature(feat2, feature_dim, feat2.shape[0])
    pcd_coord1 = make_open3d_point_cloud(kp1.numpy())
    pcd_coord2 = make_open3d_point_cloud(kp2.numpy())

    # ransac based eval
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_coord1, pcd_coord2, pcd_feat1, pcd_feat2,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist_th)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_it, 0.999))

    return ransac_result


def calculate_repeatability(kp1, kp2, T_gt, threshold: float):
    # Transform the source point cloud to the same position as the target cloud
    kp1_pos_trans = apply_transform(kp1, torch.tensor(T_gt, dtype=torch.float))
    dist = torch.cdist(kp1_pos_trans, kp2)      # (n_keypoints1, n_keypoints2) tensor

    # *** COMPUTE REPEATABILITY ***
    # Match keypoints from the first cloud with closests keypoints in the second cloud
    min_dist, _ = torch.min(dist, dim=1)
    # Repeatability with a distance threshold th
    return torch.mean((min_dist <= threshold).float()).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    parser.add_argument('--dataset_root', type=str, default='~/Data/NCLT', help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, default='nclt', choices=['mulran', 'southbay', 'kitti', 'nclt','oxford'])
    parser.add_argument('--eval_set', type=str, default='test_2012-02-04_2012-03-17_20.0_5.0.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[2, 5, 10, 20, 25], help='True Positive thresholds in meters')
    parser.add_argument('--n_k', type=int, nargs='+', default=[128], help='Number of keypoints to calculate repeatability')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weight', type=str, default=None, help='Trained global model weight')
    parser.add_argument('--icp_refine', dest='icp_refine', action='store_true')
    parser.set_defaults(icp_refine=False)
    # Ignore keypoint saliency and choose keypoint randomly
    parser.add_argument('--ignore_keypoint_saliency', dest='ignore_keypoint_saliency', action='store_true')
    parser.set_defaults(ignore_keypoint_saliency=False)
    # Ignore keypoint position and assume keypoints are located at the supervoxel centres
    parser.add_argument('--ignore_keypoint_regressor', dest='ignore_keypoint_regressor', action='store_true')
    parser.set_defaults(ignore_keypoint_regressor=False)
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for evaluation')
    
    args = parser.parse_args()
    dataset_root = os.path.expanduser(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Number of keypoints for repeatability: {args.n_k}')
    print(f'Number of sampled query elements: {args.n_samples}')
    print(f'Model config path: {args.model_config}')
    print(f'Experiment name: {args.exp_name}')
    
    if args.weight is None:
        w = 'RANDOM WEIGHT'
    else:
        w = args.weight
    print(f'Weight: {w}')
    print(f'ICP refine: {args.icp_refine}')
    print(f'Ignore keypoints saliency: {args.ignore_keypoint_saliency}')
    print(f'Ignore keypoints regressor: {args.ignore_keypoint_regressor}')
    print('')

    model_params = ModelParams(args.model_config, args.dataset_type, dataset_root)
    model_params.print()

    if args.exp_name is None:
        exp_name = model_params.model
    else:
        exp_name = args.exp_name
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model = model_factory(model_params)
    if args.weight is not None:
        assert os.path.exists(args.weight), 'Cannot open network weight: {}'.format(args.weight)
        print('Loading weight: {}'.format(args.weight))
        model.load_state_dict(torch.load(args.weight, map_location=device))

    model.to(device)

    if args.ignore_keypoint_regressor:
        model.ignore_keypoint_regressor = True

    evaluator = MinkLocGLEvaluator(dataset_root, args.dataset_type, args.eval_set, device, params=model_params, radius=args.radius,
                                   n_samples=args.n_samples, n_k=args.n_k, quantizer=model_params.quantizer,
                                   icp_refine=args.icp_refine, ignore_keypoint_saliency=args.ignore_keypoint_saliency)
    global_metrics = evaluator.evaluate(model, exp_name=exp_name)
    global_metrics['weight'] = args.weight
    evaluator.print_results(global_metrics)
    evaluator.export_eval_stats(f'./results/{exp_name}/eval_results_{args.dataset_type}.txt', global_metrics)
