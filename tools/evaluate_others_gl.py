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
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms as transforms
from glnet.models.utils import *
from glnet.config.config import *
from glnet.utils.loss_utils import *
from glnet.utils.common_utils import _ex, to_numpy, to_torch
from glnet.models.model_factory import model_factory
from glnet.datasets.range_image import range_projection
from glnet.utils.params import TrainingParams, ModelParams
from glnet.utils.data_utils.poses import m2ypr, m2xyz_ypr, xyz_ypr2m, relative_pose, relative_pose_batch
from glnet.utils.data_utils.point_clouds import generate_bev, generate_bev_occ, icp, o3d_icp, fast_gicp, make_open3d_feature, make_open3d_point_cloud
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader, get_pointcloud_with_image_loader
from tools.evaluator import Evaluator
from tools.plot_PR_curve import compute_PR_pairs
from tools.plot_pose_errors import plot_cdf, cal_recall_pe


class GLEvaluator(Evaluator):
    # Evaluation of Global Localization
    default_config = {
        'positive_threshold': 10, # estimate relative pose of two images within 10 meters
        'rotation_threshold': 5, # rotation thershold to calculation pose estimation success rate
        'translation_threshold': 2, # translation thershold to calculation pose estimation success rate
    }    
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float], k: int = 20, n_samples=None, debug: bool = False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        self.params = params

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        [model.eval() for model in models]

    def evaluate(self, model, exp_name=None, icp_refine=False, one_shot=False, config=default_config, *args, **kwargs):
        model.eval()
        if exp_name is None:
            exp_name = self.params.model
        if self.dataset_type == 'nclt':
            gt_icp_refine = False # use icp pose refinement
        elif self.dataset_type == 'oxford':
            gt_icp_refine = True # use icp pose refinement   
        
        map_images, map_embeddings, map_unet_outs, map_bevs, map_specs = self.compute_embeddings(self.eval_set.map_set, model)
        query_images, query_embeddings, query_unet_outs, query_bevs, query_specs = self.compute_embeddings(self.eval_set.query_set, model)

        map_xys = map_positions = self.eval_set.get_map_positions()
        tree = KDTree(map_positions)
        query_xys = query_positions = self.eval_set.get_query_positions()
        map_poses = self.eval_set.get_map_poses()
        query_poses = self.eval_set.get_query_poses()
        
        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')
        
        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')

        if self.n_samples is None or len(query_embeddings) <= self.n_samples:
            query_indexes = list(range(len(query_embeddings)))
            self.n_samples = len(query_embeddings)
        else:
            query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)
        
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

        for query_ndx in tqdm.tqdm(query_indexes):
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
            time_start_searching = time.time()
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            pair_dists[query_ndx, :] = embed_dist
            idxs_sorted = np.argsort(embed_dist)
            idx_top1 = idxs_sorted[0]
            time_end_searching = time.time()
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

            # Ground Truth Pose
            T_gt = relative_pose(query_pose, map_pose)
            # Ground Truth Pose Refinement
            if gt_icp_refine:
                # _, T_gt, _ = o3d_icp(query_pc[:,:3], map_pc[:,:3], transform=T_gt, point2plane=True)
                _, T_gt = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=T_gt) # spends less time than open3d
            rel_x, rel_y, rel_z, rel_yaw, rel_pitch, rel_roll = m2xyz_ypr(T_gt)
            gt_d = np.sqrt(rel_x**2 + rel_y**2)
            
            if 'disco' in self.params.model and query_unet_outs is not None:
                time_start_estimation = time.time()
                query_unet_out = query_unet_outs[query_ndx]
                query_spec = query_specs[query_ndx]
                map_unet_out = map_unet_outs[idx_top1]
                map_spec = map_specs[idx_top1]
                pred_yaw, corr = phase_corr(query_unet_out, map_unet_out)
                pred_yaw = pred_yaw.squeeze().detach().cpu().numpy() / query_unet_out.shape[-1] * 2 * np.pi  # in radians
                T_estimated = xyz_ypr2m(0, 0, 0, pred_yaw, 0, 0)
                time_end_estimation = time.time()
                time_estimation = time_end_estimation - time_start_estimation
                print(f'Pose Estimation Time of DiSCO: {time_estimation:.6f}')
            else:
                T_estimated = np.eye(4)
            if icp_refine:
                # _, T_estimated, _ = o3d_icp(query_pc[:,:3], map_pc[:,:3], transform=T_estimated, point2plane=True)
                _, T_estimated = fast_gicp(query_pc[:,:3], map_pc[:,:3], max_correspondence_distance=3.0, init_pose=T_estimated)

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
            'exp_name': exp_name, 'dataset_type': self.params.dataset_type, 'eval_setting': eval_setting,
            'topk': self.k, 'radius': self.radius, 'one_shot_gl': one_shot, 'gt_icp_refine': gt_icp_refine,
            'estimate_icp_refine': icp_refine, 'num_queries': num_queries, 'num_maps': num_maps
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


    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval((model,))
        images = None
        unet_outs = None
        bevs = None
        specs = None
        global_embeddings = None
        depths = None

        if self.params.dataset_type == 'nclt':
            pc_bev_conf = nclt_pc_bev_conf
        elif self.params.dataset_type == 'oxford':
            pc_bev_conf = oxford_pc_bev_conf
        bounds = (pc_bev_conf['x_bound'][0], pc_bev_conf['x_bound'][1], pc_bev_conf['y_bound'][0], \
                  pc_bev_conf['y_bound'][1], pc_bev_conf['z_bound'][0], pc_bev_conf['z_bound'][1])
        X = pc_bev_conf['x_grid']
        Y = pc_bev_conf['y_grid']
        Z = pc_bev_conf['z_grid']
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            if self.params.use_panorama:
                sph = True
            else:
                sph = False

            if self.params.dataset_type == 'oxford':
                extrinsics_dir = os.path.join(self.params.dataset_folder, 'extrinsics')
                scan_filepath = e.filepaths[0]
                assert os.path.exists(scan_filepath)
                orig_pc, orig_imgs = self.pcim_loader(e.filepaths, sph, extrinsics_dir)
                bev_path = scan_filepath.replace('velodyne_left', 'bev')
                bev_path = bev_path.replace('png', 'npy')
                bev_path = bev_path.replace('bin', 'npy')                
                range_image_path = bev_path.replace('bev', 'range_image')
            elif self.params.dataset_type == 'nclt':
                scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                assert os.path.exists(scan_filepath)
                orig_pc, orig_imgs = self.pcim_loader(scan_filepath, sph)
                bev_path = scan_filepath.replace('velodyne_sync', 'bev')
                bev_path = bev_path.replace('bin', 'npy')                
                range_image_path = bev_path.replace('bev', 'range_image')
            
            orig_pc = np.array(orig_pc)
            orig_imgs = np.array(orig_imgs)
            if self.params.use_bev:
                bev_folder = bev_path.strip(bev_path.split('/')[-1])
                if not os.path.exists(bev_folder):
                    os.makedirs(bev_folder)
                if os.path.isfile(bev_path):
                    bev = np.load(bev_path)
                    bev = torch.from_numpy(bev).float()
                else:
                    bev = generate_bev(orig_pc, Z=Z, Y=Y, X=X, bounds=bounds)
                    # bev = torch.from_numpy(generate_bev_occ(orig_pc, Z=Z, Y=Y, X=X, bounds=bounds))
                    np.save(bev_path, bev.numpy())
                pc = bev.unsqueeze(0).cuda()            
            elif self.params.use_range_image:
                range_image_folder = range_image_path.strip(range_image_path.split('/')[-1])
                if not os.path.exists(range_image_folder):
                    os.makedirs(range_image_folder)
                if os.path.isfile(range_image_path):
                    depths = np.load(range_image_path)
                else:
                    if orig_pc.shape[-1] == 3:
                        cloud = np.concatenate((orig_pc, np.ones((len(orig_pc), 1))), axis=1)
                    else:
                        cloud = orig_pc
                    depths, _,  _, _ = range_projection(cloud)
                    np.save(range_image_path, depths)
                depths = torch.from_numpy(depths).repeat(1, 1, 1, 1).cuda()
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
                
                lidar_data = torch.from_numpy(lidar_data).float()
                lidar_extra = torch.from_numpy(lidar_extra).float()

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
            
            if self.params.use_range_image:
                batch = {'depth': depths, 'img': imgs, 'orig_pc': orig_pc}
            else:
                batch = {'pc': pc, 'img': imgs, 'orig_pc': orig_pc}
            
            global_embedding, unet_out, bev, spec = self.compute_embedding(batch, model)

            if images is None and self.params.use_rgb:
                images = np.zeros((len(eval_subset), orig_imgs.shape[-4], orig_imgs.shape[-3], orig_imgs.shape[-2], orig_imgs.shape[-1]), dtype=orig_imgs.dtype)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[1]), dtype=global_embedding.dtype)
            if unet_outs is None and unet_out is not None:
                unet_outs = torch.zeros((len(eval_subset), unet_out.shape[-3], unet_out.shape[-2], unet_out.shape[-1])).to(unet_out)
            if bevs is None and bev is not None:
                bevs = torch.zeros((len(eval_subset), bev.shape[-3], bev.shape[-2], bev.shape[-1])).to(bev)
            if specs is None and spec is not None:
                specs = torch.zeros((len(eval_subset), spec.shape[-3], spec.shape[-2], spec.shape[-1])).to(spec)
            if self.params.use_rgb:
                images[ndx] = orig_imgs
            global_embeddings[ndx] = global_embedding
            if unet_outs is not None:
                unet_outs[ndx] = unet_out
                specs[ndx] = spec
            if bevs is not None:
                bevs[ndx] = bev
        
        return images, global_embeddings, unet_outs, bevs, specs


    def compute_embedding(self, batch, model, *args, **kwargs):
        """
        Returns global embedding (np.array)
        """
        time_start = time.time()

        with torch.no_grad():
            
            # Compute global descriptor
            y = model(batch)
            global_embedding = y['global'].detach().cpu().numpy()
            if 'unet_out' in y.keys():
                unet_out = y['unet_out']
            else:
                unet_out = None
            if 'bev' in y.keys():
                bev = y['bev']
            else:
                bev = None
            if 'spec' in y.keys():
                spec = y['spec']
            else:
                spec = None
            
        time_end = time.time()
        time_diff = time_end - time_start
        print(f'Embedding Generation Time: {time_diff:.6f}')

        return global_embedding, unet_out, bev, spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Fusion model')
    parser.add_argument('--dataset_root', type=str, default='~/Data/NCLT', help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, default='nclt', choices=['mulran', 'southbay', 'kitti', 'nclt', 'oxford'])
    parser.add_argument('--eval_set', type=str, default='test_2012-02-04_2012-03-17_20.0_5.0.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[2, 5, 10, 20, 25], help='True positive thresholds in meters')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weight', type=str, default=None, help='Trained global model weight')
    parser.add_argument('--icp_refine', dest='icp_refine', action='store_true')
    parser.set_defaults(icp_refine=False)
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for evaluation')
    parser.add_argument('--one_shot', action='store_true', help='Perform one-shot global localization without place recognition threshold limits')
    
    args = parser.parse_args()
    dataset_root = _ex(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Number of sampled query elements: {args.n_samples}')
    print(f'Model config path: {args.model_config}')
    print(f'One shot global localization: {args.one_shot}')
    
    weight = _ex(args.weight) if args.weight is not None else None
    print(f'Weight: {weight}')
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
    if weight is not None:
        assert os.path.exists(weight), 'Cannot open network weight: {}'.format(weight)
        print('Loading weight: {}'.format(weight))
        
        model = nn.DataParallel(model)
        if 'netvlad_pretrain' in model_params.model:
            checkpoint = torch.load(weight, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
        elif 'disco' in model_params.model:
            corr2soft = Corr2Softmax(200., 0.)
            checkpoint = torch.load(weight, map_location=device)
            print('checkpoint key', checkpoint.keys())
            # model.load_state_dict(checkpoint, strict=True)
            model.load_state_dict(checkpoint['model'], strict=True)
            # corr2soft.load_state_dict(checkpoint['corr2soft'], strict=True)
            corr2soft = corr2soft.to(device)
        else:
            checkpoint = torch.load(weight, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=True)

    model.to(device)
    
    with torch.no_grad():
        evaluator = GLEvaluator(dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius,
                                    n_samples=args.n_samples, params=model_params)
        global_metrics = evaluator.evaluate(model, exp_name=exp_name, icp_refine=args.icp_refine, one_shot=args.one_shot)
        global_metrics['weight'] = weight
        evaluator.print_results(global_metrics)
        evaluator.export_eval_stats(f'./results/{exp_name}/eval_results_{args.dataset_type}.txt', global_metrics)
