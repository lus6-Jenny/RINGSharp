# Warsaw University of Technology

import argparse

import numpy as np
import tqdm
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
import random
from typing import List
from time import time
import torch
import MinkowskiEngine as ME
from collections import OrderedDict

from glnet.datasets.quantization import Quantizer
from glnet.utils.data_utils.point_clouds import icp, o3d_icp, fast_gicp, make_open3d_feature, make_open3d_point_cloud
from glnet.utils.data_utils.poses import m2ypr, m2xyz_ypr, relative_pose, apply_transform

from glnet.utils.params import ModelParams
from glnet.utils.common_utils import _ex, to_numpy, to_torch
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from glnet.models.utils import *
from glnet.config.config import *

from tools.evaluator import Evaluator
from tools.plot_PR_curve import compute_PR_pairs
from tools.plot_pose_errors import plot_cdf, cal_recall_pe

from pcdet.config import cfg_from_yaml_file
from pcdet.config import cfg as pvrcnn_cfg
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

from glnet.models.backbone3D.PVRCNN import PVRCNN
from glnet.models.backbone3D.NetVlad import NetVLADLoupe
from glnet.models.backbone3D.models_3d import NetVladCustom

import open3d as o3d
if hasattr(o3d, 'pipelines'):
    reg_module = o3d.pipelines.registration
else:
    reg_module = o3d.registration

torch.backends.cudnn.benchmark = True    
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))

def get_model(exp_cfg, is_training=True):
    rotation_parameters = 1
    exp_cfg['use_svd'] = True

    if exp_cfg['3D_net'] == 'PVRCNN':
        cfg_from_yaml_file(os.path.join(parent_directory, 'glnet/models/backbone3D/pv_rcnn.yaml'), pvrcnn_cfg)
        pvrcnn_cfg.MODEL.PFE.NUM_KEYPOINTS = exp_cfg['num_points']
        if 'PC_RANGE' in exp_cfg:
            pvrcnn_cfg.DATA_CONFIG.POINT_CLOUD_RANGE = exp_cfg['PC_RANGE']
        pvrcnn = PVRCNN(pvrcnn_cfg, is_training, exp_cfg['model_norm'], exp_cfg['shared_embeddings'])
        net_vlad = NetVLADLoupe(feature_size=pvrcnn_cfg.MODEL.PFE.NUM_OUTPUT_FEATURES,
                                cluster_size=exp_cfg['cluster_size'],
                                output_dim=exp_cfg['feature_output_dim_3D'],
                                gating=True, add_norm=True, is_training=is_training)
        model = NetVladCustom(pvrcnn, net_vlad, feature_norm=False, fc_input_dim=640,
                              points_num=exp_cfg['num_points'], head=exp_cfg['head'],
                              rotation_parameters=rotation_parameters, sinkhorn_iter=exp_cfg['sinkhorn_iter'],
                              use_svd=exp_cfg['use_svd'], sinkhorn_type=exp_cfg['sinkhorn_type'])
    else:
        raise TypeError("Unknown 3D network")
    return model


class GLEvaluator(Evaluator):
    # Evaluation of Global Localization
    default_config = {
        'positive_threshold': 10, # estimate relative pose of two images within 10 meters
        'rotation_threshold': 5, # rotation thershold to calculation pose estimation success rate
        'translation_threshold': 2, # translation thershold to calculation pose estimation success rate
    }
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float], k: int = 20, n_samples=None, repeat_dist_th: float = 0.5, quantizer: Quantizer = None,
                 icp_refine: bool = True, debug: bool = False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        assert quantizer is not None
        self.repeat_dist_th = repeat_dist_th
        self.quantizer = quantizer
        self.icp_refine = icp_refine

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self, model, exp_name=None, normalize=False, ransac=False, one_shot=False, config=default_config, *args, **kwargs):
        if exp_name is None:
            exp_name = self.params.model
        if self.dataset_type == 'nclt':
            gt_icp_refine = False # use icp pose refinement
        elif self.dataset_type == 'oxford':
            gt_icp_refine = True # use icp pose refinement

        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model, normalize)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model, normalize)

        map_xys = map_positions = self.eval_set.get_map_positions()
        tree = KDTree(map_positions)
        query_xys = query_positions = self.eval_set.get_query_positions()

        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')

        metrics = {'rre': [], 'rte': [], 
                   'success': [], 'rre_refined': [], 
                   'rte_refined': [], 'success_refined': []}
        
        total_positives = np.zeros(len(self.radius)) # Total number of positive matches
        positive_threshold = config['positive_threshold'] # Positive threshold for global localization
        rotation_threshold = config['rotation_threshold'] # Rotation threshold to calculation pose estimation success rate
        translation_threshold = config['translation_threshold'] # translation thershold to calculation pose estimation success rate        
        
        quantiles = [0.25, 0.5, 0.75, 0.95]
        map_xys_retrieved = np.zeros_like(query_xys) # Top1 retrieved position

        pair_dists = np.zeros((num_queries, num_maps))
        if one_shot:
            yaw_errors = [] # rotation errors
            x_errors = [] # x translation errors
            y_errors = [] # y translation errors
            trans_errors = [] # 2D translation errors
            real_dists = [] # translation distances
        else:
            recalls_n = np.zeros((len(self.radius), self.k)) # Recall@n
            recalls_one_percent = np.zeros(len(self.radius)) # Recall@1% (recall when 1% of the database is retrieved)
            threshold = max(int(round(num_maps/100.0)), 1) # Recall@1% threshold
            yaw_errors = {r: [] for r in self.radius} # rotation errors
            x_errors = {r: [] for r in self.radius}  # x translation errors
            y_errors = {r: [] for r in self.radius}  # y translation errors
            trans_errors = {r: [] for r in self.radius}  # 2D translation errors
            real_dists = {r: [] for r in self.radius}  # translation distances

            thresholds = np.linspace(0, 1, 100) # thresholds of PR curve  
            precisions_PR_curve = np.zeros((len(self.radius), len(thresholds))) # precisions of PR curve
            recalls_PR_curve = np.zeros((len(self.radius), len(thresholds))) # recalls of PR curve
            f1s_PR_curve = np.zeros((len(self.radius), len(thresholds))) # F1 scores of PR curve

        eval_setting = self.eval_set_filepath.split('/')[-1].split('.pickle')[0]
        if one_shot:
            radius_name = 'one_shot_gl'
        else:        
            radius_name = 'revisit'
            for j, revisit_threshold in enumerate(self.radius):
                radius_name = f'{radius_name}_{revisit_threshold}'
        folder_path = _ex(f'./results/{exp_name}/{eval_setting}/{radius_name}')
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
            
            # ------------ Place Recognition ------------
            # Nearest neighbour search in the embedding space
            time_start_searching = time()
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            pair_dists[query_ndx, :] = embed_dist
            idxs_sorted = np.argsort(embed_dist)
            idx_top1 = idxs_sorted[0]
            time_end_searching = time()
            time_diff_searching = time_end_searching - time_start_searching
            print(f'NN Searching Time of Place Recognition: {time_diff_searching:.6f}')
            map_position = map_positions[idx_top1]
            map_pose = self.eval_set.map_set[idx_top1].pose
            map_xys_retrieved[query_ndx] = map_position
            if self.dataset_type == 'nclt':
                map_filepath = os.path.join(self.dataset_root, self.eval_set.map_set[idx_top1].rel_scan_filepath)
            elif self.dataset_type == 'oxford':
                map_filepath = self.eval_set.map_set[idx_top1].filepaths
            map_pc, _ = self.pcim_loader(map_filepath)

            # LOCAL DESCRIPTOR EVALUATION
            # Do the evaluation only if the nn pose is within 20 meter threshold
            # Otherwise the overlap is too small to get reasonable results
            # Ground Truth Pose
            T_gt = relative_pose(query_pose, map_pose)
            # Ground Truth Pose Refinement
            if gt_icp_refine:
                # _, T_gt, _ = o3d_icp(query_pc[:,:3], map_pc[:,:3], transform=T_gt, point2plane=True)
                _, T_gt = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=T_gt) # spends less time than open3d
            rel_x, rel_y, rel_z, rel_yaw, rel_pitch, rel_roll = m2xyz_ypr(T_gt)
            gt_d = np.sqrt(rel_x**2 + rel_y**2)
            
            time_start_estimation = time()
            T_estimated = self.compute_transformation(query_pc, map_pc, model, ransac=ransac)
            time_end_estimation = time()
            time_estimation = time_end_estimation - time_start_estimation
            print(f'Pose Estimation Time of LCDNet: {time_estimation:.6f}')
            
            if self.icp_refine:
                # T_estimated, _, _ = icp(query_pc[:,:3], map_pc[:,:3], T_estimated)
                _, T_estimated = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=T_estimated)
            
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
            
            # 2 meters and 5 degrees threshold for successful registration
            if rte > translation_threshold or rre > rotation_threshold:
                metrics['success'].append(0.)
            else:
                metrics['success'].append(1.)
                metrics['rte'].append(rte)
                metrics['rre'].append(rre)
            
            print(f'-------- Query {query_ndx+1} matched with map {idx_top1+1} --------')
            print('Ground truth translation: x: {}, y: {}, rotation: {}'.format(rel_x, rel_y, rel_yaw))
            print('Estimated translation: x: {}, y: {}, rotation: {}'.format(est_x, est_y, est_yaw))
            
            # >>>>>> One-stage Global Localization >>>>>>
            if one_shot:
                x_errors.append(x_err)
                y_errors.append(y_err)
                trans_errors.append(trans_err)
                yaw_errors.append(yaw_err)
                real_dists.append(gt_d)
            else:
                # >>>>>> Two-stages Global Localization >>>>>>
                for j, revisit_threshold in enumerate(self.radius):
                    nn_ndx = tree.query_radius(query_position.reshape(1,-1), revisit_threshold)[0]
                    if len(nn_ndx) == 0:
                        continue
                    total_positives[j] += 1
                    # Recall@n
                    for n in range(self.k):
                        # if euclid_dist[n] <= revisit_threshold:
                        if idxs_sorted[n] in nn_ndx:
                            recalls_n[j, n] += 1
                            break
                    # Recall@1%
                    if len(list(set(idxs_sorted[0:threshold]).intersection(set(nn_ndx)))) > 0:
                        recalls_one_percent[j] += 1   
                        
                    # Pose Error
                    if idx_top1 in nn_ndx:
                        x_errors[revisit_threshold].append(x_err)
                        y_errors[revisit_threshold].append(y_err)
                        trans_errors[revisit_threshold].append(trans_err)
                        yaw_errors[revisit_threshold].append(yaw_err)
                        real_dists[revisit_threshold].append(gt_d)
        
        # Calculate mean metrics
        # ------------ Results ------------ 
        global_metrics = {
            'exp_name': exp_name, 'dataset_type': self.dataset_type, 'eval_setting': eval_setting,
            'topk': self.k, 'radius': self.radius, 'one_shot_gl': one_shot, 'normalize': normalize, 'ransac': ransac,
            'gt_icp_refine': gt_icp_refine, 'estimate_icp_refine': self.icp_refine, 'num_queries': num_queries, 'num_maps': num_maps, 
        }
        scaler = MinMaxScaler()
        pair_dists = scaler.fit_transform(pair_dists.reshape(-1, 1)).ravel().reshape(pair_dists.shape)
        if one_shot:
            mean_x_error = np.mean(x_errors)
            mean_y_error = np.mean(y_errors)
            mean_trans_error = np.mean(trans_errors)
            mean_yaw_error = np.mean(yaw_errors)
            x_error_quantiles = np.quantile(x_errors, quantiles)
            y_error_quantiles = np.quantile(y_errors, quantiles)
            trans_error_quantiles = np.quantile(trans_errors, quantiles)
            yaw_error_quantiles = np.quantile(yaw_errors, quantiles) 
            success_gl = cal_recall_pe(yaw_errors, trans_errors, rot_thre=rotation_threshold, trans_thre=translation_threshold)
            eval_stat = {'success_gl': success_gl, 'mean_x_error': mean_x_error, 'mean_y_error': mean_y_error, 'mean_trans_error': mean_trans_error, 'mean_yaw_error': mean_yaw_error, 'x_error_quantiles': x_error_quantiles, 'y_error_quantiles': y_error_quantiles, 'trans_error_quantiles': trans_error_quantiles, 'yaw_error_quantiles': yaw_error_quantiles}
            global_metrics.update(eval_stat)
            print(f'Mean x error: {mean_x_error}')
            print(f'Mean y error: {mean_y_error}')
            print(f'Mean translation error: {mean_trans_error}')
            print(f'Mean yaw error: {mean_yaw_error}')
            print(f'X error quantiles: {x_error_quantiles}')
            print(f'Y error quantiles: {y_error_quantiles}')
            print(f'Translation error quantiles: {trans_error_quantiles}')
            print(f'Yaw error quantiles: {yaw_error_quantiles}')
            print(f'Success rate of global localization at {translation_threshold} m and {rotation_threshold} degrees: {success_gl}')
            
            np.save(f'{folder_path}/query_xys.npy', query_xys)
            np.save(f'{folder_path}/map_xys.npy', map_xys)
            np.save(f'{folder_path}/map_xys_retrieved.npy', map_xys_retrieved)
            np.save(f'{folder_path}/pair_dists.npy', pair_dists)
            np.save(f'{folder_path}/x_errors.npy', x_errors)
            np.save(f'{folder_path}/y_errors.npy', y_errors)
            np.save(f'{folder_path}/trans_errors.npy', trans_errors)
            np.save(f'{folder_path}/yaw_errors.npy', yaw_errors)
            np.save(f'{folder_path}/real_dists.npy', real_dists)
        else:
            for j, revisit_threshold in enumerate(self.radius):
                # ------------ Recall@n and Recall@1% ------------
                recalls_n[j] = np.cumsum(recalls_n[j])/total_positives[j]
                recalls_one_percent[j] = recalls_one_percent[j]/total_positives[j]
                print(f'-------- Revisit threshold: {revisit_threshold} m --------')
                print(f'Recall@{self.k}: {recalls_n[j]}')
                print(f'Recall@1%: {recalls_one_percent[j]}')

                # ------------ PR Curve ------------
                save_path = f'{folder_path}/precision_recall_curve_{revisit_threshold}m.pdf'
                precisions, recalls, f1s = compute_PR_pairs(pair_dists, query_xys, map_xys, thresholds=thresholds, save_path=save_path, revisit_threshold=revisit_threshold)
                precisions_PR_curve[j] = precisions
                recalls_PR_curve[j] = recalls
                f1s_PR_curve[j] = f1s
                
                # ------------ Pose Error ------------ 
                mean_x_error = np.mean(x_errors[revisit_threshold])
                mean_y_error = np.mean(y_errors[revisit_threshold])
                mean_trans_error = np.mean(trans_errors[revisit_threshold])
                mean_yaw_error = np.mean(yaw_errors[revisit_threshold])          
                x_error_quantiles = np.quantile(x_errors[revisit_threshold], quantiles)
                y_error_quantiles = np.quantile(y_errors[revisit_threshold], quantiles)
                trans_error_quantiles = np.quantile(trans_errors[revisit_threshold], quantiles)
                yaw_error_quantiles = np.quantile(yaw_errors[revisit_threshold], quantiles) 
                num_queries_pe = recalls_n[j][0] * total_positives[j]
                recall_pe = cal_recall_pe(yaw_errors[revisit_threshold], trans_errors[revisit_threshold], num_queries=num_queries_pe, rot_thre=rotation_threshold, trans_thre=translation_threshold)
                success_gl = recalls_n[j][0] * recall_pe
                eval_stat = {'num_positives': total_positives[j], 'recall_pr': recalls_n[j], 'recall_one_percent_pr': recalls_one_percent[j], 'recall_pe': recall_pe, 'success_gl': success_gl, 'mean_x_error': mean_x_error, 'mean_y_error': mean_y_error,
                            'mean_trans_error': mean_trans_error, 'mean_yaw_error': mean_yaw_error, 'x_error_quantiles': x_error_quantiles, 'y_error_quantiles': y_error_quantiles, 'trans_error_quantiles': trans_error_quantiles, 'yaw_error_quantiles': yaw_error_quantiles}
                global_metrics[f'{revisit_threshold}m'] = eval_stat
                print(f'Mean x error: {mean_x_error}')
                print(f'Mean y error: {mean_y_error}')
                print(f'Mean translation error: {mean_trans_error}')
                print(f'Mean yaw error: {mean_yaw_error}')
                print(f'X error quantiles: {x_error_quantiles}')
                print(f'Y error quantiles: {y_error_quantiles}')
                print(f'Translation error quantiles: {trans_error_quantiles}')
                print(f'Yaw error quantiles: {yaw_error_quantiles}')
                print(f'Recall at {translation_threshold} m and {rotation_threshold} degrees: {recall_pe}')
                print(f'Success rate of global localization at {translation_threshold} m and {rotation_threshold} degrees: {success_gl}')
            
            np.save(f'{folder_path}/query_xys.npy', query_xys)
            np.save(f'{folder_path}/map_xys.npy', map_xys)
            np.save(f'{folder_path}/map_xys_retrieved.npy', map_xys_retrieved)
            np.save(f'{folder_path}/pair_dists.npy', pair_dists)
            np.save(f'{folder_path}/recalls_n.npy', recalls_n)
            np.save(f'{folder_path}/precisions.npy', precisions_PR_curve)
            np.save(f'{folder_path}/recalls.npy', recalls_PR_curve)
            np.save(f'{folder_path}/f1s.npy', f1s_PR_curve)
            np.save(f'{folder_path}/x_errors.npy', dict2array(x_errors))
            np.save(f'{folder_path}/y_errors.npy', dict2array(y_errors))
            np.save(f'{folder_path}/trans_errors.npy', dict2array(trans_errors))
            np.save(f'{folder_path}/yaw_errors.npy', dict2array(yaw_errors))
            np.save(f'{folder_path}/real_dists.npy', dict2array(real_dists))
        
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
    
    
    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, normalize=False, *args, **kwargs):
        self.model2eval((model,))
        global_embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            if self.dataset_type == 'nclt':
                scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                assert os.path.exists(scan_filepath)
            elif self.dataset_type == 'oxford':
                scan_filepath = e.filepaths
                assert os.path.exists(scan_filepath[0])
            pc, _ = self.pcim_loader(scan_filepath)

            global_embedding = self.compute_global_embedding(pc, model, normalize)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[-1]), dtype=global_embedding.dtype)

            global_embeddings[ndx] = global_embedding

        return global_embeddings
    
    
    def compute_global_embedding(self, pc, model, normalize=False, *args, **kwargs):
        model.eval()
        time_start = time()
        with torch.no_grad():
            pc_list = []
            pc = to_torch(pc, device=self.device)
            if pc.shape[-1] == 3:
                pc_reflectance = torch.ones(pc.shape[0], 1)
                pc = torch.cat((pc, pc_reflectance), 1)
            pc_list.append(model.backbone.prepare_input(pc))
            model_in = KittiDataset.collate_batch(pc_list)
            
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = to_torch(val, device=self.device)
            del pc
            
            batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)
            
            global_embedding = batch_dict['out_embedding']
            
            if normalize:
                global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)
            
        time_end = time()
        time_diff = time_end - time_start
        print(f'Embedding Generation Time: {time_diff:.6f}')
                    
        return global_embedding.detach().cpu().numpy()


    def compute_transformation(self, query_pc, map_pc, model, ransac=False, *args, **kwargs):
        model.eval()
        # import pdb; pdb.set_trace()
        with torch.no_grad():
        # with torch.autocast(device_type='cuda', dtype=torch.half), torch.no_grad():
            pc_list = []
            query_pc = to_torch(query_pc, device=self.device)
            if query_pc.shape[-1] == 3:
                query_pc_reflectance = torch.ones(query_pc.shape[0], 1).to(query_pc)
                query_pc = torch.cat((query_pc, query_pc_reflectance), 1)
            map_pc = to_torch(map_pc, device=self.device)
            if map_pc.shape[-1] == 3:
                map_pc_reflectance = torch.ones(map_pc.shape[0], 1).to(query_pc)
                map_pc = torch.cat((map_pc, map_pc_reflectance), 1)
            
            pc_list.append(model.backbone.prepare_input(query_pc))
            pc_list.append(model.backbone.prepare_input(map_pc))
            model_in = KittiDataset.collate_batch(pc_list)
            
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = to_torch(val, device=self.device)
            del query_pc
            del map_pc
            
            torch.cuda.synchronize()
            batch_dict = model(model_in, metric_head=True)
            torch.cuda.synchronize()
            yaw = batch_dict['out_rotation']
            if ransac:
                coords = batch_dict['point_coords'].view(batch_dict['batch_size'], -1, 4)
                feats = batch_dict['point_features'].squeeze(-1)
                for i in range(batch_dict['batch_size'] // 2):
                    coords1 = coords[i]
                    coords2 = coords[i + batch_dict['batch_size'] // 2]
                    feat1 = feats[i]
                    feat2 = feats[i + batch_dict['batch_size'] // 2]
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector(coords1[:, 1:].cpu().numpy())
                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector(coords2[:, 1:].cpu().numpy())
                    pcd1_feat = reg_module.Feature()
                    pcd1_feat.data = feat1.permute(0, 1).cpu().numpy()
                    pcd2_feat = reg_module.Feature()
                    pcd2_feat.data = feat2.permute(0, 1).cpu().numpy()

                    torch.cuda.synchronize()
                    try:
                        result = reg_module.registration_ransac_based_on_feature_matching(
                            pcd2, pcd1, pcd2_feat, pcd1_feat, True,
                            0.6,
                            reg_module.TransformationEstimationPointToPoint(False),
                            3, [],
                            reg_module.RANSACConvergenceCriteria(5000))
                    except:
                        result = reg_module.registration_ransac_based_on_feature_matching(
                            pcd2, pcd1, pcd2_feat, pcd1_feat,
                            0.6,
                            reg_module.TransformationEstimationPointToPoint(False),
                            3, [],
                            reg_module.RANSACConvergenceCriteria(5000))

                    T_estimated = result.transformation
                    T_estimated = np.linalg.inv(T_estimated)
            else:
                transformation = batch_dict['transformation']
                homogeneous = torch.tensor([0., 0., 0., 1.]).repeat(transformation.shape[0], 1, 1).to(transformation.device)
                transformation = torch.cat((transformation, homogeneous), dim=1)
                # transformation = transformation.inverse()
                T_estimated = transformation[0].detach().cpu().numpy()
        
        # print('Estimate transformation: {}'.format(T_estimated))
        return T_estimated
    

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MinkLoc model')
    parser.add_argument('--dataset_root', type=str, default='~/Data/NCLT', help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, default='nclt', choices=['mulran', 'southbay', 'kitti', 'nclt','oxford'])
    parser.add_argument('--eval_set', type=str, default='test_2012-02-04_2012-03-17_20.0_5.0.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[2, 5, 10, 20, 25], help='True Positive thresholds in meters')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weight', type=str, required=True, help='Trained global model weight')
    parser.add_argument('--ransac', dest='ransac', action='store_true')
    parser.add_argument('--icp_refine', dest='icp_refine', action='store_true')
    parser.set_defaults(icp_refine=False)
    parser.add_argument('--one_shot', action='store_true', help='Perform one-shot global localization without place recognition threshold limits')
    # Ignore keypoint saliency and choose keypoint randomly
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for evaluation')
    
    args = parser.parse_args()
    dataset_root = _ex(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Number of sampled query elements: {args.n_samples}')
    print(f'Model config path: {args.model_config}')
    print(f'One shot global localization: {args.one_shot}')
    
    print(f'Weight: {args.weight}')
    print(f'ICP refine: {args.icp_refine}')
    print('')

    model_params = ModelParams(args.model_config, args.dataset_type, dataset_root)
    model_params.print()

    if args.exp_name is None:
        exp_name = model_params.model
    else:
        exp_name = args.exp_name
    print(f'Experiment name: {exp_name}')
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))
    
    assert os.path.exists(args.weight), 'Cannot open network weight: {}'.format(args.weight)
    print('Loading weight: {}'.format(args.weight))
    saved_params = torch.load(args.weight, map_location='cpu')
    exp_cfg = saved_params['config']
    exp_cfg['batch_size'] = 6
    exp_cfg['head'] = 'UOTHead'
    normalize = exp_cfg['norm_embeddings']
    
    model = get_model(exp_cfg, is_training=False)
    renamed_dict = OrderedDict()
    for key in saved_params['state_dict']:
        if not key.startswith('module'):
            renamed_dict = saved_params['state_dict']
            break
        else:
            renamed_dict[key[7:]] = saved_params['state_dict'][key]
    
    if renamed_dict['backbone.backbone.conv_input.0.weight'].shape != model.state_dict()['backbone.backbone.conv_input.0.weight'].shape:
        print('The weights shape is not the same as the model!')
        for key in renamed_dict:
            if key.startswith('backbone.backbone.conv') and key.endswith('weight'):
                if len(renamed_dict[key].shape) == 5:
                    renamed_dict[key] = renamed_dict[key].permute(-1, 0, 1, 2, 3) # (1, 2, 3, 4, 0)

    res = model.load_state_dict(renamed_dict, strict=True)
    
    if len(res[0]) > 0:
        print(f"WARNING: MISSING {len(res[0])} KEYS, MAYBE WEIGHTS LOADING FAILED")
    
    model = model.to(device)
    
    evaluator = GLEvaluator(dataset_root, args.dataset_type, args.eval_set, device, params=model_params, radius=args.radius,
                            n_samples=args.n_samples, quantizer=model_params.quantizer, icp_refine=args.icp_refine)
    global_metrics = evaluator.evaluate(model, exp_name=exp_name, normalize=normalize, ransac=args.ransac, one_shot=args.one_shot)
    global_metrics['weight'] = args.weight
    evaluator.print_results(global_metrics)
    evaluator.export_eval_stats(f'./results/{exp_name}/eval_results_{args.dataset_type}.txt', global_metrics)
