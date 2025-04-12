# Zhejiang University

import argparse
import numpy as np
import tqdm
import os
import cv2
import time
import random
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as transforms
from glnet.models.backbones_2d.unet import last_conv_block
from glnet.models.model_factory import model_factory
from glnet.utils.data_utils.point_clouds import generate_bev, o3d_icp, fast_gicp
from glnet.utils.data_utils.poses import apply_transform, m2ypr, m2xyz_ypr, xyz_ypr2m, relative_pose, relative_pose_batch
from glnet.utils.params import ModelParams
from glnet.utils.common_utils import _ex, to_numpy, to_torch
from glnet.datasets.panorama import generate_sph_image
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader, get_pointcloud_with_image_loader
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from glnet.models.utils import *
from glnet.config.config import *
from glnet.models.voting import (
    make_grid,
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
    TemplateSampler,
)
from tools.evaluator import Evaluator
from tools.plot_PR_curve import compute_PR_pairs
from tools.plot_pose_errors import plot_cdf, cal_recall_pe
from tools.viz_loc import imshow, transform_point_cloud, plot_point_cloud, plot_matched_point_clouds, plot_heatmap

class GLEvaluator(Evaluator):
    # Evaluation of Global Localization
    default_config = {
        'positive_threshold': 10, # estimate relative pose of two images within 10 meters
        'rotation_threshold': 5, # rotation thershold to calculation pose estimation success rate
        'translation_threshold': 2, # translation thershold to calculation pose estimation success rate
    }   
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float], k: int=20, n_samples=None, debug: bool=False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        self.params = params
        if dataset_type == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif dataset_type == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        self.bounds = (pc_bev_conf['x_bound'][0], pc_bev_conf['x_bound'][1], pc_bev_conf['y_bound'][0], \
                  pc_bev_conf['y_bound'][1], pc_bev_conf['z_bound'][0], pc_bev_conf['z_bound'][1])
        self.X = pc_bev_conf['x_grid']
        self.Y = pc_bev_conf['y_grid']
        self.Z = pc_bev_conf['z_grid']
        
    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models
        [model.eval() for model in models]

    def evaluate(self, model, trans_cnn, exp_name=None, n_rerank=1000, refine=True, icp_refine=False, one_shot=False, do_viz=False, config=default_config):
        if exp_name is None:
            exp_name = self.params.model
        if self.dataset_type == 'nclt':
            gt_icp_refine = False # use icp pose refinement
            pc_bev_conf = nclt_pc_bev_conf
        elif self.dataset_type == 'oxford':
            gt_icp_refine = True # use icp pose refinement
            pc_bev_conf = oxford_pc_bev_conf
        do_argmax = True
        compute_query_embeddings = False
        
        # load embedding model
        self.model2eval((model, trans_cnn))
        map_pcs, map_images, map_embeddings, map_bevs, map_specs, map_bevs_trans = self.compute_embeddings(self.eval_set.map_set, model, trans_cnn, phase='map')
        if compute_query_embeddings:
            query_pcs, query_images, query_embeddings, query_bevs, query_specs, query_bevs_trans = self.compute_embeddings(self.eval_set.query_set, model, trans_cnn, phase='query')
        
        Ns, Cs, Hs, Ws = map_specs.shape
        Nt, Ct, Ht, Wt = map_bevs_trans.shape

        map_xys = map_positions = self.eval_set.get_map_positions()
        tree = KDTree(map_positions)
        query_xys = query_positions = self.eval_set.get_query_positions()
        map_poses = self.eval_set.get_map_poses()
        query_poses = self.eval_set.get_query_poses()

        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')
        n_rerank = min(n_rerank, num_maps)

        batch_size = 128
        if self.params.use_rgb:
            x_range = backbone_conf['x_bound'][1] - backbone_conf['x_bound'][0]
            y_range = backbone_conf['y_bound'][1] - backbone_conf['y_bound'][0]
            ppm = 1 / backbone_conf['x_bound'][2] # pixel per meter   
        else:
            x_range = pc_bev_conf['x_bound'][1] - pc_bev_conf['x_bound'][0]
            y_range = pc_bev_conf['y_bound'][1] - pc_bev_conf['y_bound'][0]
            ppm = Ht / x_range # pixel per meter
        template_sampler = TemplateSampler(Hs, optimize=False) # template matching
        
        total_positives = np.zeros(len(self.radius)) # Total number of positive matches
        positive_threshold = config['positive_threshold'] # Positive threshold for global localization
        rotation_threshold = config['rotation_threshold'] # Rotation threshold to calculation pose estimation success rate
        translation_threshold = config['translation_threshold'] # translation thershold to calculation pose estimation success rate        
        
        quantiles = [0.25, 0.5, 0.75, 0.95]
        map_xys_retrieved = np.zeros_like(query_xys) # Top1 retrieved position

        pair_dists = np.zeros((num_queries, num_maps))
        query_est_xys = [] # estimated query positions
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
        name = ''
        if n_rerank > 0:
            if n_rerank == num_maps:
                name += f'rerank_all'
            else:
                name += f'rerank_{n_rerank}'
        if refine:
            name = f'{name}_refine'
        if name == '':
            folder_path = _ex(f'./results/{exp_name}/{eval_setting}/{radius_name}')
        else:
            folder_path = _ex(f'./results/{exp_name}/{eval_setting}/{radius_name}/{name}')
        os.makedirs(folder_path, exist_ok=True)

        # for query_ndx in tqdm.tqdm(range(num_queries)):
        for query_ndx, eval_data in tqdm.tqdm(enumerate(self.eval_set.query_set)):
            # Check if the query element has a true match within each radius
            if compute_query_embeddings:
                query_pc = query_pcs[query_ndx]
                if self.params.use_rgb:
                    query_image = query_images[query_ndx]
                query_bev = query_bevs[query_ndx]
                query_spec = query_specs[query_ndx]
                if query_bevs_trans is not None:
                    query_bev_trans = query_bevs_trans[query_ndx]
                else:
                    query_bev_trans = None
            else:
                query_pc, query_image, query_embedding, query_bev, query_spec, query_bev_trans = self.compute_embedding_item(eval_data, model)
            C, H, W = query_bev.shape[-3:]
            query_position = query_positions[query_ndx]
            query_pose = query_poses[query_ndx]

            # ------------ Place Recognition ------------
            if num_maps >= batch_size and num_maps % batch_size == 0:
                batch_num = num_maps // batch_size
            else:
                batch_num = num_maps // batch_size + 1   
            
            time_start_rank = time.time()  
            for i in range(batch_num):
                if i == batch_num - 1:
                    map_spec = [map_specs[k] for k in range(i*batch_size, num_maps)]
                else:
                    map_spec = [map_specs[k] for k in range(i*batch_size, (i+1)*batch_size)]
                map_spec = torch.stack(map_spec, dim=0).reshape((-1, Cs, Hs, Ws))
                query_spec_repeated = query_spec.repeat(map_spec.shape[0], 1, 1, 1)

                # time_start = time.time()
                batch_corrs, batch_scores, batch_shifts = estimate_yaw(query_spec_repeated, map_spec)
                batch_angles = batch_shifts * 2 * np.pi / Hs
                batch_corrs = batch_corrs.detach().cpu().numpy()
                batch_scores = batch_scores.detach().cpu().numpy()
                batch_angles = batch_angles.detach().cpu().numpy()
                # time_end = time.time()
                # print('Time of batchwise rotation estimation', time_end - time_start)                
                if i == 0:
                    # corrs = batch_corrs
                    scores = batch_scores
                    angles = batch_angles
                else:
                    # corrs = np.concatenate((corrs, batch_corrs), axis=-1)
                    scores = np.concatenate((scores, batch_scores), axis=-1)
                    angles = np.concatenate((angles, batch_angles), axis=-1)
            time_end_rank = time.time()
            time_diff_rank = time_end_rank - time_start_rank
            print(f'Initial Ranking Time of RING#-{n_rerank} in Rotation Branch: {time_diff_rank:.6f}')
            
            # corrs = corrs.squeeze()
            dists = 1 - scores.squeeze()
            angles = angles.squeeze()    
            pair_dists[query_ndx] = dists
            dists_sorted = np.sort(dists)
            idxs_sorted = np.argsort(dists)

            if n_rerank > 0:
                # Rerank top n matches by yaw BEV features
                idxs_topn = idxs_sorted[:n_rerank]
                errors_topn = []
                pred_x_topn = []
                pred_y_topn = []
                pred_yaw_topn = []
                corrs_topn = []
                query_bev_trans_rot_topn = []

                if n_rerank >= batch_size and n_rerank % batch_size == 0:
                    batch_num = n_rerank // batch_size
                else:
                    batch_num = n_rerank // batch_size + 1  
                
                time_start_rerank = time.time()
                for i in range(batch_num):
                    if i == batch_num - 1:
                        batch_idxs = [idxs_topn[k] for k in range(i*batch_size, n_rerank)]
                    else:
                        batch_idxs = [idxs_topn[k] for k in range(i*batch_size, (i+1)*batch_size)]                            
                    batch_angles = angles[batch_idxs]
                    batch_angles_extra = batch_angles - np.pi  
                    if map_bevs_trans is not None:
                        map_bev_trans = map_bevs_trans[batch_idxs]
                    else:
                        map_bev_trans = trans_cnn(map_bevs[batch_idxs])
                        
                    if query_bev_trans is not None:
                        query_bev_trans_rotated = rotate_bev_batch(query_bev_trans, array2tensor(batch_angles))
                        query_bev_trans_rotated_extra = rotate_bev_batch(query_bev_trans, array2tensor(batch_angles_extra))
                    else:
                        query_bev_rotated = rotate_bev_batch(query_bev, array2tensor(batch_angles))
                        query_bev_rotated_extra = rotate_bev_batch(query_bev, array2tensor(batch_angles_extra))                
                        query_bev_trans_rotated = trans_cnn(query_bev_rotated)
                        query_bev_trans_rotated_extra = trans_cnn(query_bev_rotated_extra)
                                    
                    # time_start = time.time()
                    x, y, errors, corrs = solve_translation(query_bev_trans_rotated, map_bev_trans)
                    x_extra, y_extra, errors_extra, corrs_extra = solve_translation(query_bev_trans_rotated_extra, map_bev_trans)
                    errors = errors.detach().cpu().numpy()
                    errors_extra = errors_extra.detach().cpu().numpy()
                    # time_end = time.time()
                    # print('Time of batchwise translation estimation', time_end - time_start) 

                    errors_topn = errors_topn + [errors[i] if errors[i] < errors_extra[i] else errors_extra[i] for i in range(len(batch_idxs))]                
                    pred_x_topn = pred_x_topn + [x[i] / Ht * x_range if errors[i] < errors_extra[i] else x_extra[i] / Ht * x_range for i in range(len(batch_idxs))]  # in meters
                    pred_y_topn = pred_y_topn + [y[i] / Wt * y_range if errors[i] < errors_extra[i] else y_extra[i] / Wt * y_range for i in range(len(batch_idxs))]  # in meters
                    pred_yaw_topn = pred_yaw_topn + [batch_angles[i] if errors[i] < errors_extra[i] else batch_angles_extra[i] for i in range(len(batch_idxs))]  # in radians
                    corrs_topn = corrs_topn + [corrs[i] if errors[i] < errors_extra[i] else corrs_extra[i] for i in range(len(batch_idxs))] # correlation map
                    query_bev_trans_rot_topn = query_bev_trans_rot_topn + [query_bev_trans_rotated[i] if errors[i] < errors_extra[i] else query_bev_trans_rotated_extra[i] for i in range(len(batch_idxs))]
                time_end_rerank = time.time()
                time_diff_rerank = time_end_rerank - time_start_rerank
                time_searching = time_end_rerank - time_start_rank
                print(f'Reranking Time of RING#-{n_rerank} in Translation Branch: {time_diff_rerank:.6f}')
                print(f'Searching Time of RING#-{n_rerank}: {time_searching:.6f}')
                
                errors_topn = np.array(errors_topn)
                pred_x_topn = np.array(pred_x_topn)
                pred_y_topn = np.array(pred_y_topn)
                pred_yaw_topn = np.array(pred_yaw_topn)
                corrs_topn = torch.stack(corrs_topn, dim=0)
                query_bev_trans_rot_topn = torch.stack(query_bev_trans_rot_topn, dim=0)
                
                idxs_errors = np.argsort(errors_topn)
                idxs_topn = idxs_topn[idxs_errors]
                idxs_sorted[:n_rerank] = idxs_topn                
                errors_topn = errors_topn[idxs_errors]
                pred_x_topn = pred_x_topn[idxs_errors]
                pred_y_topn = pred_y_topn[idxs_errors]
                pred_yaw_topn = pred_yaw_topn[idxs_errors]
                corrs_topn = corrs_topn[idxs_errors]
                query_bev_trans_rot_topn = query_bev_trans_rot_topn[idxs_errors]
                if n_rerank == num_maps:
                    pair_dists[query_ndx] = errors_topn[np.argsort(idxs_topn)]
                    
            idx_top1 = idxs_sorted[0]
            
            # ------------ Pose Estimation ------------
            # Perform pose estimation for the top 1 match
            map_pc = map_pcs[idx_top1]
            if self.params.use_rgb:
                map_image = map_images[idx_top1]
            map_position = map_positions[idx_top1]
            map_xys_retrieved[query_ndx] = map_position            
            map_pose = map_poses[idx_top1]            
            map_spec = map_specs[idx_top1]
            map_bev_trans = map_bevs_trans[idx_top1]
            
            # Ground Truth Pose
            rel_pose = relative_pose(query_pose, map_pose)
            # Ground Truth Pose Refinement
            if gt_icp_refine:
                # _, rel_pose, _ = o3d_icp(query_pc[:,:3], map_pc[:,:3], transform=rel_pose, point2plane=True)
                _, rel_pose = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=rel_pose) # spends less time than open3d
            raw_x, raw_y, raw_z, raw_yaw, raw_pitch, raw_roll = m2xyz_ypr(rel_pose)
            if self.params.use_rgb:
                gt_yaw = -raw_yaw # in radians
                gt_x = raw_y # in meters
                gt_y = raw_x # in meters
            else:
                gt_yaw = raw_yaw # in radians
                gt_x = raw_x # in meters
                gt_y = raw_y # in meters                
            gt_d = np.sqrt(gt_x**2 + gt_y**2) # in meters
            gt_x_grid = Ht / 2 - np.ceil(gt_x * Ht / x_range) # in grids
            gt_y_grid = Wt / 2 - np.ceil(gt_y * Wt / y_range) # in grids
            gt_yaw_grid = Ht / 2 - np.ceil(gt_yaw * Ht / (2 * np.pi)) # in grids
            
            if refine:
                time_start_refine = time.time()
                if query_bev_trans is not None:
                    templates = template_sampler(query_bev_trans)
                else:
                    templates = trans_cnn(template_sampler(query_bev).reshape(-1, C, H, W)).reshape(1, -1, Ct, Ht, Wt)
                scores = conv2d_fft_batchwise(
                    map_bev_trans.unsqueeze(0).float(),
                    templates.float(),
                )
                scores = scores.moveaxis(1, -1)  # B,H,W,N
                scores = torch.fft.fftshift(scores, dim=(-3, -2))
                log_probs = log_softmax_spatial(scores)
                with torch.no_grad():
                    uvr_max = argmax_xyr(scores).to(scores)
                    uvr_avg, _ = expectation_xyr(log_probs.exp())
                
                if do_argmax:
                    # Argmax
                    pred_x = uvr_max[..., 0].detach().cpu().numpy() / ppm # in meters
                    pred_y = uvr_max[..., 1].detach().cpu().numpy() / ppm # in meters
                    pred_yaw = uvr_max[..., 2].detach().cpu().numpy() * np.pi / 180. # in radians
                    query_bev_trans_rot = templates[:, torch.round(uvr_max[..., 2] * Hs / 360).long(), ...].squeeze() 
                else:
                    # Average
                    pred_x = uvr_avg[..., 0].detach().cpu().numpy() / ppm # in meters
                    pred_y = uvr_avg[..., 1].detach().cpu().numpy() / ppm # in meters
                    pred_yaw = uvr_avg[..., 2].detach().cpu().numpy() * np.pi / 180. # in radians  
                    query_bev_trans_rot = templates[:, torch.round(uvr_avg[..., 2] * Hs / 360).long(), ...].squeeze()  
                time_end_refine = time.time()
                time_diff_refine = time_end_refine - time_start_refine
                print(f'Refining Time of RING#-{n_rerank} in Translation Branch: {time_diff_refine:.6f}')
                corr = corrs_topn[0] # correlation map
            elif n_rerank > 0:
                pred_x = pred_x_topn[0] # in meters
                pred_y = pred_y_topn[0] # in meters
                pred_yaw = pred_yaw_topn[0] # in radians
                corr = corrs_topn[0] # correlation map
                query_bev_trans_rot = query_bev_trans_rot_topn[0]
            else:
                angle = angles[idx_top1]
                angle_extra = angle - np.pi
                if query_bevs_trans is not None:
                    query_bev_trans = query_bevs_trans[query_ndx]
                    query_bev_trans_rotated = rotate_bev_batch(query_bev_trans, array2tensor(angle))
                    query_bev_trans_rotated_extra = rotate_bev_batch(query_bev_trans, array2tensor(angle_extra))
                else:
                    query_bev_rotated = rotate_bev_batch(query_bev, array2tensor(angle))
                    query_bev_rotated_extra = rotate_bev_batch(query_bev, array2tensor(angle_extra))                
                    query_bev_trans_rotated = trans_cnn(query_bev_rotated)
                    query_bev_trans_rotated_extra = trans_cnn(query_bev_rotated_extra)
            
                x, y, error, corr = solve_translation(query_bev_trans_rotated, map_bev_trans)
                x_extra, y_extra, error_extra, corr_extra = solve_translation(query_bev_trans_rotated_extra, map_bev_trans)
                if error < error_extra:
                    pred_x = x.squeeze() / Ht * x_range # in meters
                    pred_y = y.squeeze() / Wt * y_range # in meters 
                    pred_yaw = angle # in radians
                    query_bev_trans_rot = query_bev_trans_rotated
                else:
                    pred_x = x_extra.squeeze() / Ht * x_range # in meters
                    pred_y = y_extra.squeeze() / Wt * y_range # in meters
                    pred_yaw = angle_extra # in radians
                    corr = corr_extra # correlation map
                    query_bev_trans_rot = query_bev_trans_rotated_extra
            
            if self.params.use_rgb:
                T_estimated = xyz_ypr2m(pred_y, pred_x, 0, -pred_yaw, 0, 0)
            else:
                T_estimated = xyz_ypr2m(pred_x, pred_y, 0, pred_yaw, 0, 0)
            if icp_refine:
                time_start_icp = time.time()
                # T_estimated, _, _ = icp(query_pc[:,:3], map_pc[:,:3], T_estimated)
                _, T_estimated = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=T_estimated)
                pred_x, pred_y, _, pred_yaw, _, _ = m2xyz_ypr(T_estimated)
                time_end_icp = time.time()
                time_diff_icp = time_end_icp - time_start_icp
                print(f'ICP Refinement Time: {time_diff_icp:.6f}')
            
            query_est_T = map_pose @ T_estimated
            query_est_x, query_est_y, _, query_est_yaw, _, _ = m2xyz_ypr(query_est_T)
            query_est_xys.append([query_est_x, query_est_y])
            x_err = np.abs(gt_x - pred_x) # in meters
            y_err = np.abs(gt_y - pred_y) # in meters
            trans_err = np.sqrt(x_err**2 + y_err**2) # in meters
            yaw_err = np.abs(angle_clip(gt_yaw - pred_yaw)) * 180 / np.pi # in degrees
            
            if query_bev_trans is None:
                query_bev_trans = trans_cnn(query_bev)
            
            query_bev_trans = query_bev_trans.squeeze().detach().cpu().numpy()
            map_bev_trans = map_bev_trans.squeeze().detach().cpu().numpy()
            if query_bev_trans.ndim == 3:
                # Sum
                query_bev_trans = query_bev_trans.sum(axis=0)
                map_bev_trans = map_bev_trans.sum(axis=0)
            gt_xy_grid = [gt_x_grid, gt_y_grid]
            
            if do_viz:
                # Successful Case (5deg, 2m)
                if yaw_err < 5 and trans_err < 2: # np.abs(gt_yaw) > 1 and gt_d > 5 and 
                    save_folder_path = f'{folder_path}/succ_query_{query_ndx}_map_{idx_top1}_{gt_yaw}_{gt_x}_{gt_y}_{gt_d}_{yaw_err}_{x_err}_{y_err}_{trans_err}'
                    os.makedirs(save_folder_path, exist_ok=True)
                    if self.params.use_rgb:
                        sensor = 'vision'
                        query_sph_img = generate_sph_image(query_image, self.dataset_type, self.dataset_root)
                        map_sph_img = generate_sph_image(map_image, self.dataset_type, self.dataset_root)
                        if self.dataset_type == 'nclt':
                            query_sph_img = cv2.cvtColor(query_sph_img, cv2.COLOR_RGB2BGR)
                            map_sph_img = cv2.cvtColor(map_sph_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f'{save_folder_path}/query_sph.jpg', query_sph_img)
                        cv2.imwrite(f'{save_folder_path}/map_sph.jpg', map_sph_img)
                        for n in range(query_image.shape[0]):
                            query_image_n = query_image[n]
                            positive_map_image_n = map_image[n]
                            cv2.imwrite(f'{save_folder_path}/query_cam_{n+1}.jpg', query_image_n)
                            cv2.imwrite(f'{save_folder_path}/map_cam_{n+1}.jpg', positive_map_image_n)
                    else:
                        sensor = 'lidar'
                        plot_point_cloud(query_pc, s=0.3, c='g', alpha=0.5, save_path=f'{save_folder_path}/query_pc.jpg')
                        plot_point_cloud(map_pc, s=0.3, c='b', alpha=0.5, save_path=f'{save_folder_path}/map_pc.jpg')
                        query_pc_transformed = transform_point_cloud(query_pc, T_estimated)
                        plot_matched_point_clouds(query_pc_transformed, map_pc, s=0.3, alpha=0.5, save_path=f'{save_folder_path}/matched_pc.jpg', transpose=True)
                    
                    imshow(query_bev_trans.squeeze(), f'{save_folder_path}/query_bev_trans.jpg')
                    imshow(map_bev_trans.squeeze(), f'{save_folder_path}/map_bev_trans.jpg')
                    
                    if gt_d < 100:
                        plot_heatmap(corr, gt_xy_grid, gt_yaw, None, pred_yaw, f'{save_folder_path}/heatmap.jpg', sensor=sensor)
                
                # Failure Case (5deg, 2m)
                if yaw_err > 5 or trans_err > 2:
                    save_folder_path = f'{folder_path}/fail_query_{query_ndx}_map_{idx_top1}_{gt_yaw}_{gt_x}_{gt_y}_{gt_d}_{yaw_err}_{x_err}_{y_err}_{trans_err}'
                    os.makedirs(save_folder_path, exist_ok=True)
                    if self.params.use_rgb:
                        sensor = 'vision'
                        query_sph_img = generate_sph_image(query_image, self.dataset_type, self.dataset_root)
                        map_sph_img = generate_sph_image(map_image, self.dataset_type, self.dataset_root)
                        if self.dataset_type == 'nclt':
                            query_sph_img = cv2.cvtColor(query_sph_img, cv2.COLOR_RGB2BGR)
                            map_sph_img = cv2.cvtColor(map_sph_img, cv2.COLOR_RGB2BGR)                            
                        cv2.imwrite(f'{save_folder_path}/query_sph.jpg', query_sph_img)
                        cv2.imwrite(f'{save_folder_path}/map_sph.jpg', map_sph_img)
                        for n in range(query_image.shape[0]):
                            query_image_n = query_image[n]
                            positive_map_image_n = map_image[n]
                            cv2.imwrite(f'{save_folder_path}/query_cam_{n+1}.jpg', query_image_n)
                            cv2.imwrite(f'{save_folder_path}/map_cam_{n+1}.jpg', positive_map_image_n)
                    else:
                        sensor = 'lidar'
                        plot_point_cloud(query_pc, s=0.3, c='g', alpha=0.5, save_path=f'{save_folder_path}/query_pc.jpg')
                        plot_point_cloud(map_pc, s=0.3, c='b', alpha=0.5, save_path=f'{save_folder_path}/map_pc.jpg')
                        query_pc_transformed = transform_point_cloud(query_pc, T_estimated)
                        plot_matched_point_clouds(query_pc_transformed, map_pc, s=0.3, alpha=0.5, save_path=f'{save_folder_path}/matched_pc.jpg', transpose=True)
                        # query_bev_trans = query_bev_trans.squeeze().transpose(1, 0)
                        # map_bev_trans = map_bev_trans.squeeze().transpose(1, 0)
                    
                    imshow(query_bev_trans.squeeze(), f'{save_folder_path}/query_bev_trans.jpg')
                    imshow(map_bev_trans.squeeze(), f'{save_folder_path}/map_bev_trans.jpg')
                    
                    if gt_d < 100:
                        plot_heatmap(corr, gt_xy_grid, gt_yaw, None, pred_yaw, f'{save_folder_path}/heatmap.jpg', sensor=sensor)
            
            print(f'-------- Query {query_ndx+1} matched with map {idx_top1+1} --------')
            print('Ground truth translation: x: {}, y: {}, rotation: {}'.format(gt_x, gt_y, gt_yaw))
            print('Estimated translation: x: {}, y: {}, rotation: {}'.format(pred_x, pred_y, pred_yaw))
            
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
                        if idxs_sorted[n] in nn_ndx:
                            recalls_n[j, n] += 1
                            break
                    # Recall@1%
                    if len(list(set(idxs_sorted[0:threshold]).intersection(set(nn_ndx)))) > 0:
                        recalls_one_percent[j] += 1

                    # Pose Error
                    if idx_top1 in nn_ndx:
                        print(f'>>>>>>>> Successful Case of Place Recognition at {revisit_threshold} m >>>>>>>>')
                        x_errors[revisit_threshold].append(x_err)
                        y_errors[revisit_threshold].append(y_err)
                        trans_errors[revisit_threshold].append(trans_err)
                        yaw_errors[revisit_threshold].append(yaw_err)    
                        real_dists[revisit_threshold].append(gt_d)
                    else:
                        print(f'>>>>>>>> Failure Case of Place Recognition at {revisit_threshold} m >>>>>>>>')
                        # print(f'-------- Query {query_ndx+1} matched with map {idx_top1+1} --------')
                        # print('Ground truth translation: x: {}, y: {}, rotation: {}'.format(gt_x, gt_y, gt_yaw))
                        # print('Estimated translation: x: {}, y: {}, rotation: {}'.format(pred_x, pred_y, pred_yaw))

        # ------------ Results ------------
        global_metrics = {'exp_name': exp_name, 'dataset_type': self.params.dataset_type, 'eval_setting': eval_setting, 'topk': self.k, 'radius': self.radius, 'n_rerank': n_rerank, 'one_shot_gl': one_shot,
                          'refine': refine, 'gt_icp_refine': gt_icp_refine, 'do_argmax': do_argmax, 'estimate_icp_refine': icp_refine, 'quantiles': quantiles, 'num_queries': num_queries, 'num_maps': num_maps, 'time_rank': time_diff_rank}
        if n_rerank > 0:
            global_metrics['time_rerank'] = time_diff_rerank
        if refine:
            global_metrics['time_refine'] = time_diff_refine
        # scaler = MinMaxScaler()
        # pair_dists = scaler.fit_transform(pair_dists.reshape(-1, 1)).ravel().reshape(pair_dists.shape)
        np.save(f'{folder_path}/query_est_xys.npy', query_est_xys)
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
    
    
    def compute_embedding_item(self, eval_data, model, trans_cnn=None):
        if self.params.use_panorama:
            sph = True
        else:
            sph = False

        if self.params.dataset_type == 'oxford':
            extrinsics_dir = os.path.join(self.params.dataset_folder, 'extrinsics')
            scan_filepath = eval_data.filepaths[0]
            # assert os.path.exists(scan_filepath)
            orig_pc, orig_imgs = self.pcim_loader(eval_data.filepaths, sph, extrinsics_dir)
            bev_path = scan_filepath.replace('velodyne_left', 'bev')
            bev_path = bev_path.replace('png', 'npy')
            bev_path = bev_path.replace('bin', 'npy')  
        elif self.params.dataset_type == 'nclt':
            scan_filepath = os.path.join(self.dataset_root, eval_data.rel_scan_filepath)
            assert os.path.exists(scan_filepath)
            orig_pc, orig_imgs = self.pcim_loader(scan_filepath, sph)
            bev_path = scan_filepath.replace('velodyne_sync', 'bev')
            bev_path = bev_path.replace('bin', 'npy')
        
        orig_pc = np.array(orig_pc)
        orig_imgs = np.array(orig_imgs)
        if self.params.use_bev:
            bev_folder = bev_path.strip(bev_path.split('/')[-1])
            os.makedirs(bev_folder, exist_ok=True)
            if os.path.isfile(bev_path):
                bev = np.load(bev_path)
                bev = to_torch(bev)
            else:
                bev = generate_bev(orig_pc, Z=self.Z, Y=self.Y, X=self.X, bounds=self.bounds)
                np.save(bev_path, bev.numpy())
            pc = bev.unsqueeze(0).cuda()
        else:
            V = self.params.lidar_fix_num
            lidar_data = orig_pc[:,:3]
            lidar_extra = orig_pc[:,3:]
            if orig_pc.shape[0] > V:
                lidar_data = lidar_data[:V]
                lidar_extra = lidar_extra[:V]
            elif orig_pc.shape[0] < V:
                lidar_data = np.pad(lidar_data,[(0,V-lidar_data.shape[0]),(0,0)],mode='constant')
                lidar_extra = np.pad(lidar_extra,[(0,V-lidar_extra.shape[0]),(0,0)],mode='constant',constant_values=-1)
            
            lidar_data = to_torch(lidar_data)
            lidar_extra = to_torch(lidar_extra)

            pc = torch.cat([lidar_data, lidar_extra], dim=1)   
            pc = pc.unsqueeze(0)

            if self.params.quantizer == None:
                pc = pc
            else:
                pc = self.params.quantizer(pc)
            pc = pc.cuda()

        toTensor = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        if self.params.use_rgb:
            imgs = [toTensor(e) for e in orig_imgs]
            imgs = torch.stack(imgs).float().cuda()
            if not sph:
                imgs = imgs.unsqueeze(0).cuda()
        else:
            imgs = None
        
        batch = {'pc': pc, 'img': imgs, 'orig_pc': orig_pc}
        x, bev, spec, bev_trans = self.compute_embedding(batch, model)
        if bev_trans is None and trans_cnn is not None:
            bev_trans = trans_cnn(bev)
        
        return orig_pc, orig_imgs, x, bev, spec, bev_trans
    
            
    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, trans_cnn, phase='map'):
        pcs = []
        images = None
        embeddings = None
        bevs = None
        specs = None
        bevs_trans = None
        
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            if phase == 'map':
                orig_pc, orig_imgs, x, bev, spec, bev_trans = self.compute_embedding_item(e, model, trans_cnn)
            else:
                orig_pc, orig_imgs, x, bev, spec, bev_trans = self.compute_embedding_item(e, model)

            if self.params.use_rgb and images is None:
                images = np.zeros((len(eval_subset), orig_imgs.shape[-4], orig_imgs.shape[-3], orig_imgs.shape[-2], orig_imgs.shape[-1]), dtype=orig_imgs.dtype)
            # if embeddings is None:
            #     embeddings = torch.zeros((len(eval_subset), x.shape[-2], x.shape[-1])).to(x.device)
            if bevs is None and phase == 'query':
                bevs = torch.zeros((len(eval_subset), bev.shape[-3], bev.shape[-2], bev.shape[-1])).to(bev.device)
            if specs is None:
                specs = torch.zeros((len(eval_subset), spec.shape[-3], spec.shape[-2], spec.shape[-1])).to(spec.device)
            
            if bev_trans is not None:
                if bevs_trans is None:
                    bevs_trans = torch.zeros((len(eval_subset), bev_trans.shape[-3], bev_trans.shape[-2], bev_trans.shape[-1])).to(bev_trans.device)  
                bevs_trans[ndx] = bev_trans
            
            pcs.append(orig_pc[:, :3])
            if self.params.use_rgb:
                images[ndx] = orig_imgs
            # embeddings[ndx] = x
            if phase == 'query':
                bevs[ndx] = bev
            specs[ndx] = spec

        return pcs, images, embeddings, bevs, specs, bevs_trans


    def compute_embedding(self, batch, model):
        '''
        Returns BEV features
        '''
        time_start = time.time()

        with torch.no_grad():
            y = model(batch)           
            x = y['global']
            bev = y['bev']        
            spec = y['spec']
            bev_trans = y['bev_trans']            

        time_end = time.time()
        time_diff = time_end - time_start
        print(f'Embedding Generation Time: {time_diff:.6f}')  

        return x, bev, spec, bev_trans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Fusion model')
    parser.add_argument('--dataset_root', type=str, default='~/Data/NCLT', help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, default='nclt', choices=['kitti', 'nclt', 'oxford'])
    parser.add_argument('--eval_set', type=str, default='test_2012-02-04_2012-03-17_20.0_5.0.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[2, 5, 10, 20, 25], help='True positive thresholds in meters')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weight', type=str, default=None, help='Trained global model weight')
    parser.add_argument('--icp_refine', dest='icp_refine', action='store_true')
    parser.set_defaults(icp_refine=False)
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for evaluation')
    parser.add_argument('--n_rerank', type=int, default=1, help='Rerank top n matches retrieved by yaw BEV features')
    parser.add_argument('--refine', action='store_true', help='Refine the estimated pose by 3-Dof exhaustive matching')
    parser.add_argument('--one_shot', action='store_true', help='Perform one-shot global localization without place recognition threshold limit')
    parser.add_argument('--viz', action='store_true', help='Visualize the results')
    
    args = parser.parse_args()
    dataset_root = _ex(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Model config path: {args.model_config}')
    print(f'Rerank top {args.n_rerank} matches retrieved by yaw BEV features')
    print(f'Refine: {args.refine}')
    print(f'One shot global localization: {args.one_shot}')
    print(f'Visualization: {args.viz}')
    
    if args.weight is None:
        w = 'RANDOM WEIGHT'
    else:
        w = args.weight
    print(f'Weight: {w}')
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
        device = 'cuda'
    else:
        device = 'cpu'
    print('Device: {}'.format(device))

    model = model_factory(model_params)
    model.print_info()
    
    if model_params.use_rgb:
        feature_dim = backbone_conf['output_channels']
        trans_cnn = nn.Sequential(
                nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=feature_dim, out_channels=1, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
    else:
        if model_params.use_bev:
            feature_dim = model_params.feature_dim
            mid_channels = model_params.feature_dim
        else:
            feature_dim = model_params.feature_dim
            mid_channels = feature_dim
        trans_cnn = last_conv_block(feature_dim, mid_channels, bn=False)
    trans_cnn = trans_cnn.to(device)
    
    if args.weight is not None:
        assert os.path.exists(args.weight), 'Cannot open network weight: {}'.format(args.weight)
        print('Loading weight: {}'.format(args.weight))
        
        if device == 'cpu':
            pass
        else:
            model = nn.DataParallel(model)
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        trans_cnn.load_state_dict(checkpoint['trans_cnn'], strict=False)

    model.to(device)
    
    with torch.no_grad():
        evaluator = GLEvaluator(dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius, params=model_params)
        time_start = time.time()
        global_metrics = evaluator.evaluate(model, trans_cnn, exp_name=exp_name, n_rerank=args.n_rerank, refine=args.refine, icp_refine=args.icp_refine, one_shot=args.one_shot, do_viz=args.viz)
        global_metrics['weight'] = args.weight
        time_end = time.time()
        print(f'Total time of RING#-{args.n_rerank}: ', time_end-time_start)
        evaluator.print_results(global_metrics)
        evaluator.export_eval_stats(f'./results/{exp_name}/eval_results_{args.dataset_type}.txt', global_metrics)
