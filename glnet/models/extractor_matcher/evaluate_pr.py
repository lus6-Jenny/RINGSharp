# Zhejiang University


import os
import time
import tqdm
import random
import pickle
import argparse
import numpy as np
from typing import List
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler

import cv2
import torch
import torch.nn as nn
import open3d as o3d
import matplotlib.cm as cm
from torchvision import transforms as transforms
from glnet.utils.data_utils.point_clouds import o3d_icp, fast_gicp
from glnet.utils.data_utils.poses import apply_transform, m2ypr, m2xyz_ypr, relative_pose
from glnet.utils.params import TrainingParams, ModelParams
import glnet.utils.vox_utils.basic as basic
import glnet.utils.vox_utils.geom as geom
from glnet.utils.common_utils import _ex, to_numpy, to_torch

from glnet.datasets.nclt.nclt_raw import pc2image_file
from glnet.datasets.nclt.project_vel_to_cam_rot import project_vel_to_cam, project_vel_to_cam_oxford
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader, get_pointcloud_with_image_loader
from glnet.datasets.panorama import generate_sph_image
from glnet.models.utils import *
from glnet.config.config import *
from glnet.models.localizer.ring_sharp_v import RINGSharpV
from glnet.models.extractor_matcher.matching import Matching 
from glnet.models.extractor_matcher.match_features_demo import preprocess_image, match_descriptors
from glnet.models.extractor_matcher.utils import remove_none_values, remove_at_indices, assign_matched_cams, convert_to_3d, scale_intrinsics, estimate_pose, solve_pnp, estimate_pose_ransac_3d3d, make_matching_plot
from tools.evaluator import Evaluator
from tools.plot_PR_curve import compute_PR_pairs
from tools.plot_pose_errors import plot_cdf, cal_recall_pe


class GLEvaluator(Evaluator):     
    """Evaluation of Pose Estimation by Feature Extraction and Matching Methods
    Feature Extrator: SIFT, ORB, SuperPoint, etc.
    Feature Matcher: NearestNeighbor, SuperGlue, LoFTR, etc.
    """  
    default_config = {
        'positive_threshold': 10, # estimate relative pose of two images within 10 meters
        'rotation_threshold': 5, # rotation thershold to calculation pose estimation success rate
        'translation_threshold': 2, # translation thershold to calculation pose estimation success rate
    }
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float], k: int = 20, n_samples = None, debug: bool = False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        # self.params = params
        image_meta_path = os.path.expanduser(self.params.image_meta_path)
        with open(image_meta_path, 'rb') as handle:
            image_meta = pickle.load(handle)
        __p = lambda x: basic.pack_seqdim(x, 1)
        __u = lambda x: basic.unpack_seqdim(x, 1)            
        self.K = np.array(image_meta['K'])    
        self.num_cams = self.K.shape[0]   
        print(f"Number of cameras: {self.num_cams}")
        self.T = np.array(image_meta['T'])
        cams_T_body = torch.from_numpy(self.T).unsqueeze(0).float()
        cams_T_body = cams_T_body.repeat(1,1,1,1).cuda()
        body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
        camXs_T_camXs = []
        for i in range(self.num_cams):
            cami_T_camXs = __p(geom.get_camM_T_camXs(body_T_cams, ind=i))
            camXs_T_camXs.append(cami_T_camXs)
        self.cams_T_body = cams_T_body.detach().cpu().numpy().squeeze()
        self.body_T_cams = body_T_cams.detach().cpu().numpy().squeeze()
        self.camXs_T_camXs = torch.stack(camXs_T_camXs).detach().cpu().numpy().squeeze()


    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]


    def evaluate(self, model_matching, model_depth=None, extractor="superpoint", matcher="superglue", exp_name=None, resize=[-1], do_viz=False, do_pnp=True, one_shot=False, config=default_config):
        if exp_name is None:
            exp_name = f'{extractor}_{matcher}_{self.dataset_type}'
        gt_depths = True # use ground truth depths from 3D point clouds
        do_3d = False # use 3d-3d ransac pose solver
        if self.dataset_type == 'nclt':
            gt_icp_refine = False # use icp pose refinement
        elif self.dataset_type == 'oxford':
            gt_icp_refine = True # use icp pose refinement
        
        map_image_paths, map_pcs, map_images, map_depths = self.load_images(self.eval_set.map_set, model_depth=model_depth, gt_depths=gt_depths)
        query_image_paths, query_pcs, query_images, query_depths = self.load_images(self.eval_set.query_set, model_depth=model_depth, gt_depths=gt_depths)

        # GTs
        map_xys = map_positions = self.eval_set.get_map_positions()
        tree = KDTree(map_positions)
        query_xys = query_positions = self.eval_set.get_query_positions()
        map_poses = self.eval_set.get_map_poses()
        query_poses = self.eval_set.get_query_poses()
        
        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')

        if self.n_samples is None or num_queries <= self.n_samples:
            query_indexes = list(range(num_queries))
            self.n_samples = num_queries
        else:
            query_indexes = random.sample(range(num_queries), self.n_samples)

        x_errors = []
        y_errors = []
        trans_errors = []
        yaw_errors = []
        real_dists = []

        thresh = 1.
        if do_pnp:
            min_matches = 4 # Minimum number of matches for PnP
        else:
            min_matches = 5 # Minimum number of matches for essential matrix calculation
        total_positives = np.zeros(len(self.radius)) # Total number of positive matches
        positive_threshold = config['positive_threshold'] # Positive threshold for pose estimation
        rotation_threshold = config['rotation_threshold'] # Rotation threshold to calculation pose estimation success rate
        translation_threshold = config['translation_threshold'] # Translation thershold to calculation pose estimation success rate  
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
        folder_path = os.path.expanduser(f'./results/{exp_name}/{eval_setting}/{radius_name}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        pair_dists = np.zeros((num_queries, num_maps))
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pc = query_pcs[query_ndx]
            query_depth = query_depths[query_ndx]
            query_pos = query_positions[query_ndx]
            query_pose = query_poses[query_ndx]
            qyaw, qpitch, qroll = m2ypr(query_pose)
            file_pathname = query_image_paths[query_ndx]
                        
            if self.dataset_type == 'nclt':
                query_image_path = [pc2image_file(file_pathname, '/velodyne_sync/', i, '.bin') for i in range(1, 6)]
            elif self.dataset_type == 'oxford':
                query_image_path = [file_pathname[i] for i in range(2, 6)]
            query_orig, query_gray, query_norm, query_scale = preprocess_image(self.dataset_type, query_image_path, resize=resize)

            if do_viz:
                save_path = f"{folder_path}/query_{query_ndx}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            
            xs = []
            ys = []
            yaws = []
            Ts = []
            scores = np.zeros(num_maps)
            for idx in range(num_maps):
                map_pc = map_pcs[idx]
                map_depth = map_depths[idx]
                map_pose = map_poses[idx]
                file_pathname = map_image_paths[idx]
                if self.dataset_type == 'nclt':
                    map_image_path = [pc2image_file(file_pathname, '/velodyne_sync/', i, '.bin') for i in range(1, 6)]
                elif self.dataset_type == 'oxford':
                    map_image_path = [file_pathname[i] for i in range(2, 6)]
                
                # Ground Truth Pose
                rel_pose = relative_pose(query_pose, map_pose)
                # Ground Truth Pose Refinement
                if gt_icp_refine:
                    # _, rel_pose, _ = o3d_icp(query_pc[:,:3], map_pc[:,:3], transform=rel_pose, point2plane=True)   
                    _, rel_pose = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=rel_pose) # spends less time than open3d             
                rel_x, rel_y, rel_z, rel_yaw, rel_pitch, rel_roll = m2xyz_ypr(rel_pose)
                t_scale = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
                gt_d = np.sqrt(rel_x**2 + rel_y**2)
                shift_cam = assign_matched_cams(rel_yaw, self.num_cams)
                
                map_orig, map_gray, map_norm, map_scale = preprocess_image(self.dataset_type, map_image_path, resize=resize)
                
                estimate_xs = []
                estimate_ys = []
                estimate_yaws = []
                max_inliers = 0
                time_start = time.time()
                
                x = y = yaw = 0
                T = np.eye(4)
                for j in range(self.num_cams):
                    # Do not consider camera 2 of nclt dataset (containing GPS objects in the image)
                    if self.dataset_type == 'nclt' and j == 1:
                        continue
                    k = np.mod(j+shift_cam, self.num_cams)
                    img1_gray = query_gray[j]
                    img2_gray = map_gray[k]
                    img1_norm = query_norm[j]
                    img2_norm = map_norm[k]            
                    K1 = self.K[j]
                    K2 = self.K[k]   
                    K1 = scale_intrinsics(K1, query_scale)
                    K2 = scale_intrinsics(K2, map_scale)    
                                     
                    if extractor == 'sift':
                        data = {'image0': img1_gray, 'image1': img2_gray, 'extractor': extractor, 'matcher': matcher}  
                    else:
                        data = {'image0': img1_norm, 'image1': img2_norm, 'extractor': extractor, 'matcher': matcher}  
                                            
                    kp1, kp2, m_kp1, m_kp2, matches, conf = match_descriptors(model_matching, data) 

                    num_matches = 0 if matches is None else len(matches)
                    if num_matches < min_matches:
                        print(f'No. matches is {num_matches} less than {min_matches}')
                        continue
                    else:
                        # import pdb; pdb.set_trace()
                        try:
                            if do_pnp:
                                # Convert to 3D keypoints
                                img1_depth = query_depth[j]
                                m_kp1_3d = [convert_to_3d(m_kp1[i], img1_depth, K1) for i in range(len(m_kp1))]
                                indices1, values1 = remove_none_values(m_kp1_3d)
                                print(f'The number of keypoints decreases from {len(m_kp1)} to {len(values1)} since there are {len(indices1)} non-depth points of the query image!')
                                img2_depth = map_depth[k] 
                                m_kp2_3d = [convert_to_3d(m_kp2[i], img2_depth, K2) for i in range(len(m_kp2))]
                                indices2, values2 = remove_none_values(m_kp2_3d)
                                print(f'The number of keypoints decreases from {len(m_kp2)} to {len(values2)} since there are {len(indices2)} non-depth points of the query image!')                            
                                indices = indices1 + indices2
                                if do_3d:
                                    # Solve 3D-3D RANSAC
                                    m_kp1_3d_new = remove_at_indices(m_kp1_3d, indices)
                                    m_kp2_3d_new = remove_at_indices(m_kp2_3d, indices)
                                    m_kp1_new = np.array([value for i, value in enumerate(m_kp1) if i not in indices])
                                    m_kp2_new = np.array([value for i, value in enumerate(m_kp2) if i not in indices])
                                    conf_new = np.array([value for i, value in enumerate(conf) if i not in indices])                             
                                    R, t, inliers = estimate_pose_ransac_3d3d(m_kp1_3d_new, m_kp2_3d_new)
                                else:
                                    # Solve 2D-3D PnP + RANSAC
                                    m_kp1_3d_new = values1
                                    m_kp1_new = np.array([value for i, value in enumerate(m_kp1) if i not in indices1])
                                    m_kp2_new = np.array([value for i, value in enumerate(m_kp2) if i not in indices1])
                                    conf_new = np.array([value for i, value in enumerate(conf) if i not in indices1]) 
                                    R, t, inliers = solve_pnp(m_kp1_3d_new, m_kp2_new, K2)
                                num_matches = len(m_kp1_3d_new)   
                            else:
                                # Estimate pose from essential matrix
                                R, t, inliers = estimate_pose(m_kp1, m_kp2, K1, K2, thresh)
                                # Scale translation to match the ground truth
                                t = t * t_scale   
                            
                            if inliers is None:
                                print('Inliers is None!')
                                num_inliers = 0
                                continue
                            else:
                                num_inliers = len(inliers)
                    
                            est_T = np.eye(4)
                            est_T[:3, :3] = np.array(R).squeeze()
                            est_T[:3, 3] = np.array(t).squeeze()

                            camj_T_camk = self.camXs_T_camXs[j][k]
                            est_T = camj_T_camk @ est_T
                            est_T = self.body_T_cams[j] @ est_T @ self.cams_T_body[j]

                            est_x, est_y, est_z, est_yaw, est_pitch, est_roll = m2xyz_ypr(est_T)

                            # print(f'---------- Matches_query_{query_ndx}_cam_{j+1}_map_{idx}_cam_{k+1} ----------')
                            # print('GT & EST x', rel_x, est_x)
                            # print('GT & EST y', rel_y, est_y)
                            # print('GT & EST yaw', rel_yaw, est_yaw)
                            # print('GT pose', rel_x, rel_y, rel_z, rel_yaw, rel_pitch, rel_roll)
                            # print(f'Estimated pose between cam {j+1} and cam {k+1}', est_x, est_y, est_z, est_yaw, est_pitch, est_roll)     
                            # print(f'---------- Matches_query_{query_ndx}_cam_{j+1}_map_{idx}_cam_{k+1} ----------')  
                            
                            if do_viz:
                                # Visualize the matches
                                color = cm.jet(conf)
                                text = [
                                    exp_name,
                                    'Keypoints: {}:{}'.format(len(kp1), len(kp2)),
                                    'Matches: {}'.format(num_matches),
                                    'Inliers: {}'.format(num_inliers),
                                ]                  
                                            
                                # Display extra parameter info
                                if extractor == 'superpoint' and matcher == 'superglue':
                                    k_thresh = model_matching.superpoint.config['keypoint_threshold']
                                    m_thresh = model_matching.superglue.config['match_threshold']
                                    small_text = [
                                        'Keypoint Threshold: {:.4f}'.format(k_thresh),
                                        'Match Threshold: {:.2f}'.format(m_thresh),
                                    ]
                                else:
                                    small_text = []                              
                                viz_path = f'{save_path}/query_{query_ndx}_cam_{j+1}_map_{idx}_cam_{k+1}.jpg'
                                
                                if do_pnp:
                                    color_new = cm.jet(conf_new)
                                    make_matching_plot(
                                        img1_gray, img2_gray, kp1, kp2, m_kp1_new, m_kp2_new, color_new,
                                        text, viz_path, show_keypoints=True,
                                        fast_viz=False, opencv_display=False, opencv_title='Matches', small_text=small_text)    
                                else:
                                    make_matching_plot(
                                        img1_gray, img2_gray, kp1, kp2, m_kp1, m_kp2, color,
                                        text, viz_path, show_keypoints=True,
                                        fast_viz=False, opencv_display=False, opencv_title='Matches', small_text=small_text)                                      
                                                   
                            # estimate_xs.append(est_x)
                            # estimate_ys.append(est_y)
                            # estimate_yaws.append(est_yaw)
                            if num_inliers > max_inliers:
                                x = est_x
                                y = est_y
                                yaw = est_yaw
                                T = est_T
                                max_inliers = num_inliers
                        except:
                            print('Error in pose estimation!')
                            continue
                
                time_end = time.time()
                print('Multi Image Feature Extraction and Matching Time: ', time_end - time_start)
                
                # if x is None:
                #     print('Cannot estimate the pose!')
                #     continue
                
                # x = np.mean(estimate_xs)
                # y = np.mean(estimate_ys)
                # yaw = np.mean(estimate_yaws)
                # print('GT x y yaw', rel_x, rel_y, rel_yaw)
                # print('Estimated x y yaw', x, y, yaw)
                scores[idx] = max_inliers
                xs.append(x)
                ys.append(y)
                yaws.append(yaw)
                Ts.append(T)
            
            dists = 1 - scores / 128
            pair_dists[query_ndx, :] = dists
            idxs_sorted = np.argsort(dists)
            idx_top1 = idxs_sorted[0]
            map_pc = map_pcs[idx_top1]
            map_position = map_positions[idx_top1]
            map_xys_retrieved[query_ndx] = map_position             
            x = xs[idx_top1]
            y = ys[idx_top1]
            yaw = yaws[idx_top1]
            T = Ts[idx_top1]
            # Ground Truth Pose
            map_pose_matched = map_poses[idx_top1]
            rel_pose = relative_pose(query_pose, map_pose_matched)
            query_est_T = map_pose_matched @ T
            query_est_x, query_est_y, _, query_est_yaw, _, _ = m2xyz_ypr(query_est_T)
            query_est_xys.append([query_est_x, query_est_y])
            # Ground Truth Pose Refinement
            if gt_icp_refine:
                # _, rel_pose, _ = o3d_icp(query_pc[:,:3], map_pc[:,:3], transform=rel_pose, point2plane=True)
                _, rel_pose = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=rel_pose) # spends less time than open3d
            rel_x, rel_y, rel_z, rel_yaw, rel_pitch, rel_roll = m2xyz_ypr(rel_pose)
            t_scale = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
            gt_d = np.sqrt(rel_x**2 + rel_y**2)
            x_err = np.abs((rel_x - x))
            y_err = np.abs((rel_y - y))
            trans_err = np.sqrt(x_err**2 + y_err**2)
            yaw_err = np.abs((angle_clip(rel_yaw - yaw))*180/np.pi)
            
            print(f'-------- Query {query_ndx} matched with map {idx_top1} --------')
            print(f'Number of max inliers: {np.max(scores)}')
            print('Ground truth translation: x: {}, y: {}, rotation: {}'.format(rel_x, rel_y, rel_yaw))
            print('Estimated translation: x: {}, y: {}, rotation: {}'.format(x, y, yaw))
            
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
                    nn_ndx = tree.query_radius(query_pos.reshape(1,-1), revisit_threshold)[0]
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

        # ------------ Results ------------ 
        global_metrics = {
            'exp_name': exp_name, 'dataset_type': self.params.dataset_type, 'eval_setting': eval_setting, 'topk': self.k, 
            'radius': self.radius, 'do_pnp': do_pnp, 'do_3d': do_3d, 'gt_depths': gt_depths, 'one_shot_gl': one_shot, 'gt_icp_refine': gt_icp_refine, 
            'quantiles': quantiles, 'num_queries': num_queries, 'num_maps': num_maps}
        scaler = MinMaxScaler()
        pair_dists = scaler.fit_transform(pair_dists.reshape(-1, 1)).ravel().reshape(pair_dists.shape)
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


    def load_images(self, eval_subset: List[EvaluationTuple], model_depth=None, gt_depths=False):
        total_filepaths = []
        total_pcs = []
        total_images = []
        total_depth_maps = []
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            if self.params.use_panorama:
                sph = True
            else:
                sph = False

            if self.params.dataset_type == 'oxford':
                extrinsics_dir = os.path.join(self.params.dataset_folder, 'extrinsics')
                scan_filepath = e.filepaths
                orig_pc, orig_imgs = self.pcim_loader(scan_filepath, sph, extrinsics_dir)
            elif self.params.dataset_type == 'nclt':
                scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                assert os.path.exists(scan_filepath)
                orig_pc, orig_imgs = self.pcim_loader(scan_filepath, sph)  

            if gt_depths:
                S, H, W, C = np.array(orig_imgs).shape
                depth_maps = []
                for cam_num in range(S):
                    hits_pc = np.concatenate([orig_pc[:, :3], torch.ones(orig_pc.shape[0], 1)], axis=1).astype(np.float32)
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

                    depth_map = np.zeros([H, W])
                    depth_map[y_im_int, x_im_int] = z_im   
                    depth_maps.append(depth_map)    
            else:
                toTensor = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                imgs = [toTensor(e) for e in orig_imgs]
                imgs = torch.stack(imgs).float().cuda()
                if not sph:
                    imgs = imgs.unsqueeze(0).cuda()                
                depth_maps = self.pred_depth(orig_pc, imgs, model_depth)

            total_filepaths.append(scan_filepath)
            total_pcs.append(orig_pc) 
            total_images.append(orig_imgs)  
            total_depth_maps.append(depth_maps)        
        return np.array(total_filepaths), np.array(total_pcs), np.array(total_images), np.array(total_depth_maps)


    def pred_depth(self, pc, imgs, model):
        with torch.no_grad():
            batch = {'pc': pc, 'img': imgs}
            y = model(batch)           
            depth_preds = y['depth']
            B, S, C, H, W = imgs.shape
            N, D, Hd, Wd = depth_preds.shape                            
            depth_channels = int((backbone_conf['d_bound'][1] - backbone_conf['d_bound'][0]) / backbone_conf['d_bound'][2])    
            feat_depth = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, depth_channels)     
            feat_depth = (feat_depth * torch.arange(0, depth_channels, device="cuda")[None, None, :]).sum(2)
            feat_depth = (feat_depth + 0.5) * backbone_conf['d_bound'][2] + backbone_conf['d_bound'][0]        
            feat_depth = feat_depth.view(B, S, Hd, Wd)
            # feat_depth = depth_preds.view(B, S, D, Hd, Wd)
            # feat_depth = torch.max(feat_depth, dim=2)[1]         
            feat_depth = F.interpolate(feat_depth, size=(H, W), mode='bilinear', align_corners=False).view(B, S, H, W)             
            feat_depth = feat_depth.squeeze(0).detach().cpu().numpy()       

        return feat_depth   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Fusion model')
    parser.add_argument('--dataset_root', type=str, default='~/Data/NCLT', help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, default='nclt', choices=['mulran', 'southbay', 'kitti', 'nclt','oxford'])
    parser.add_argument('--eval_set', type=str, default='test_2012-02-04_2012-03-17_20.0_5.0.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[2, 5, 10, 20, 25], help='True Positive thresholds in meters')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, help='Path to the global model configuration file')
    parser.add_argument('--weight_depth', type=str, default=None, help='Trained depth model weight')
    parser.add_argument('--extractor', type=str, default='superpoint', help='Evaluated extractor')
    parser.add_argument('--matcher', type=str, default='superglue', help='Evaluated matcher')
    parser.add_argument('--positive_threshold', type=float, default=10, help='Positive Threshold for Evaluation')
    parser.add_argument('--resize', type=int, nargs='+', default=[-1], help='Resize the input image before running inference. '
                        'If two numbers, resize to the exact dimensions, if one number, resize the max dimension, if -1, do not resize')    
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for evaluation')
    parser.add_argument('--viz', action='store_true', help='Visualize the matches and dump the plots')
    parser.add_argument('--pnp', action='store_true', help='Convert the 2D keypoints of map images to 3D points')
    parser.add_argument('--one_shot', action='store_true', help='Perform one-shot global localization without place recognition threshold limits')
    
    args = parser.parse_args()
    dataset_root = os.path.expanduser(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Number of sampled query elements: {args.n_samples}')
    print(f'Model config path: {args.model_config}')
    print(f'Evaluated extractor: {args.extractor}')
    print(f'Evaluated matcher: {args.matcher}')
    print(f'One shot global localization: {args.one_shot}')
    
    model_params = ModelParams(args.model_config, args.dataset_type, dataset_root)
    model_params.print()

    if args.exp_name is None:
        pr_method = model_params.model
        exp_name = f'{args.extractor}_{args.matcher}_{args.dataset_type}'        
    else:
        exp_name = args.exp_name
    print(f'Experiment name: {exp_name}')
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print('Device: {}'.format(device))

    model_depth = RINGSharpV(model_params)
    model_depth.print_info()

    if args.weight_depth is not None:
        assert os.path.exists(args.weight_depth), 'Cannot open network weight: {}'.format(args.weight_depth)
        print('Loading weight: {}'.format(args.weight_depth))
        model_depth = nn.DataParallel(model_depth)
        checkpoint = torch.load(args.weight_depth, map_location=device)
        model_depth.load_state_dict(checkpoint['model'], strict=False)

    model_depth.to(device)

    config = {
        'sift': {
            'grayscale': True,
            'resize_max': 1600,
            'trim_edges': [16, -40],
        },
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
            'trim_edges': [16, -40],
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        },
        'nn-ratio': {
            'ratio_threshold': 0.8, 
            'distance_threshold': 0.7,
        },
        'nn-dis': {
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
        'nn-mutual': {
            'do_mutual_check': True,       
        }
    }
    matching = Matching(config).eval().to(device)

    eval_config = {
        'positive_threshold': args.positive_threshold, # estimate relative pose of two images within 10 meters
        'rotation_threshold': 5, # rotation thershold to calculation pose estimation success rate
        'translation_threshold': 2, # translation thershold to calculation pose estimation success rate
    }

    evaluator = GLEvaluator(dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius, n_samples=args.n_samples, params=model_params)
    global_metrics = evaluator.evaluate(matching, model_depth, args.extractor, args.matcher, exp_name=exp_name, resize=args.resize, do_viz=args.viz, do_pnp=args.pnp, one_shot=args.one_shot, config=eval_config)
    global_metrics['weight_depth'] = args.weight_depth
    evaluator.print_results(global_metrics)
    evaluator.export_eval_stats(f'./results/{exp_name}/eval_results_{args.dataset_type}.txt', global_metrics)