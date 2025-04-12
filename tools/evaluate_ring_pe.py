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
from torchvision import transforms as transforms
from glnet.models.utils import *
from glnet.config.config import *
from glnet.utils.loss_utils import *
from glnet.models.backbones_2d.unet import last_conv_block, UNet, Autoencoder, TransNet
from glnet.models.model_factory import model_factory
from glnet.utils.params import ModelParams
from glnet.datasets.panorama import generate_sph_image
from glnet.utils.common_utils import _ex, to_numpy, to_torch
from glnet.utils.data_utils.point_clouds import o3d_icp, fast_gicp
from glnet.utils.data_utils.poses import m2ypr, m2xyz_ypr, xyz_ypr2m, relative_pose, relative_pose_batch
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader, get_pointcloud_with_image_loader
from tools.evaluator import Evaluator
from tools.plot_pose_errors import plot_cdf, plot_yaw_errors, plot_trans_yaw_errors, cal_recall_pe
from glnet.models.loss import Max_P_2DLoss
from tools.viz_loc import imshow, plot_point_cloud, plot_heatmap, features_to_RGB


class GLEvaluator(Evaluator):
    # Evaluation of Pose Estimation
    default_config = {
        'positive_threshold': 10, # estimate relative pose of two images within 10 meters
        'rotation_threshold': 5, # rotation thershold to calculation pose estimation success rate
        'translation_threshold': 2, # translation thershold to calculation pose estimation success rate
    }    
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float], k: int=20, n_samples=None, debug: bool=False):
        super().__init__(dataset_root, dataset_type, eval_set_pickle, device, params, radius, k, n_samples, debug=debug)
        self.params = params

    def model2eval(self, models):
        # This method may be overloaded when model is a tuple consisting of a few models
        [model.eval() for model in models]

    def evaluate(self, model, trans_cnn, exp_name=None, icp_refine=True, do_viz=False, config=default_config):
        if exp_name is None:
            exp_name = self.params.model
        if self.dataset_type == 'nclt':
            gt_icp_refine = False # use icp pose refinement
            pc_bev_conf = nclt_pc_bev_conf
        elif self.dataset_type == 'oxford':
            gt_icp_refine = True # use icp pose refinement
            pc_bev_conf = oxford_pc_bev_conf
                        
        # load embedding model
        map_pcs, map_images, map_bevs, map_specs, map_embeddings, map_bevs_trans = self.compute_embeddings(self.eval_set.map_set, model, trans_cnn, phase='map')
        query_pcs, query_images, query_bevs, query_specs, query_embeddings, query_bevs_trans = self.compute_embeddings(self.eval_set.query_set, model, trans_cnn, phase='query')

        N, C, H, W = map_bevs.shape
        Ns, Cs, Hs, Ws = map_specs.shape
        if map_bevs_trans is not None:
            Nt, Ct, Ht, Wt = map_bevs_trans.shape

        map_positions = self.eval_set.get_map_positions()
        tree = KDTree(map_positions)
        query_positions = self.eval_set.get_query_positions()
        map_poses = self.eval_set.get_map_poses()
        query_poses = self.eval_set.get_query_poses()

        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')
        batch_size = 16
        if self.params.use_rgb:
            x_range = backbone_conf['x_bound'][1] - backbone_conf['x_bound'][0]
            y_range = backbone_conf['y_bound'][1] - backbone_conf['y_bound'][0]
        else:
            x_range = pc_bev_conf['x_bound'][1] - pc_bev_conf['x_bound'][0]
            y_range = pc_bev_conf['y_bound'][1] - pc_bev_conf['y_bound'][0]

        # Pose Estimation Parameters
        scale = 1 # Scale interpolation factor
        total_positives = 0 # Total number of positives        
        positive_threshold = config['positive_threshold'] # Positive threshold for pose estimation
        rotation_threshold = config['rotation_threshold'] # Rotation threshold to calculation pose estimation success rate
        translation_threshold = config['translation_threshold'] # Translation thershold to calculation pose estimation success rate        
        quantiles = [0.25, 0.5, 0.75, 0.95]
        kl_map = init_kl_map(H, 1)
        softmax = nn.Softmax(dim=-1)
        
        eval_setting = self.eval_set_filepath.split('/')[-1].split('.pickle')[0]
        folder_path = os.path.expanduser(f'./results/{exp_name}/{eval_setting}/pe_{positive_threshold}')
        os.makedirs(folder_path, exist_ok=True)
        
        for query_ndx in tqdm.tqdm(range(num_queries)):
            # Check if the query element has a true match within each radius
            query_pc = query_pcs[query_ndx]
            if self.params.use_rgb:
                query_image = query_images[query_ndx]
            query_bev = query_bevs[query_ndx]
            query_position = query_positions[query_ndx]
            query_pose = query_poses[query_ndx]
            query_spec = query_specs[query_ndx]

            nn_ndx = tree.query_radius(query_position.reshape(1,-1), positive_threshold)[0]
            # nn_ndx = np.array([1,2,3,4])
            num_positives = len(nn_ndx)
            if num_positives == 0: 
                continue
            total_positives += num_positives
            
            if self.params.use_rgb:
                positive_images = map_images[nn_ndx]
            positive_pcs = [map_pcs[i] for i in nn_ndx]
            positive_poses = map_poses[nn_ndx]
            positive_bevs = map_bevs[nn_ndx]
            positive_bevs_specs = map_specs[nn_ndx]
            positive_bevs_trans = map_bevs_trans[nn_ndx]
            query_pose = torch.from_numpy(query_pose)
            query_pose_repeated = query_pose.repeat(num_positives, 1, 1)
            positive_poses = torch.from_numpy(positive_poses)
            rel_poses = relative_pose_batch(query_pose_repeated, positive_poses)
            # Ground Truth Pose Refinement
            if gt_icp_refine:
                rel_poses_refine = []
                for i, rel_pose in enumerate(rel_poses):
                    # _, rel_pose, _ = o3d_icp(query_pc[:,:3], positive_pcs[i][:,:3], transform=rel_pose.detach().cpu().numpy(), point2plane=True)   
                    _, rel_pose = fast_gicp(query_pc[:,:3], positive_pcs[i][:,:3], init_pose=rel_pose.detach().cpu().numpy()) # spends less time than open3d  
                rel_poses_refine.append(rel_pose)
                rel_poses = to_torch(rel_poses_refine)
            raw_x, raw_y, raw_z, raw_yaw, raw_pitch, raw_roll = m2xyz_ypr(rel_poses)
            if self.params.use_rgb:
                gt_yaw = -raw_yaw # in radians
                gt_x = raw_y # in meters
                gt_y = raw_x # in meters                            
            else:
                gt_yaw = raw_yaw # in radians
                gt_x = raw_x # in meters
                gt_y = raw_y # in meters
            gt_x_grid = (H / 2 - torch.ceil(gt_x * H / x_range)).cuda() # in grids
            gt_y_grid = (W / 2 - torch.ceil(gt_y * W / y_range)).cuda() # in grids              
            gt_yaw_grid = H / 2 - np.ceil(gt_yaw * H / (2 * np.pi)) # in grids
            gt_d = np.sqrt(gt_x**2 + gt_y**2) # in meters

            # ------------ Pose Estimation ------------
            if num_positives % batch_size == 0:
                batch_num = num_positives // batch_size
            else:
                batch_num = num_positives // batch_size + 1      
            for i in range(batch_num):
                if i == batch_num - 1:
                    map_spec = [positive_bevs_specs[k] for k in range(i*batch_size, num_positives)]
                    if map_bevs_trans is not None:   
                        map_bev_trans = [positive_bevs_trans[k] for k in range(i*batch_size, num_positives)]
                    else:
                        map_bev = [positive_bevs[k] for k in range(i*batch_size, num_positives)]
                else:
                    map_spec = [positive_bevs_specs[k] for k in range(i*batch_size, (i+1)*batch_size)]
                    if map_bevs_trans is not None:
                        map_bev_trans = [positive_bevs_trans[k] for k in range(i*batch_size, (i+1)*batch_size)]
                    else:
                        map_bev = [positive_bevs[k] for k in range(i*batch_size, (i+1)*batch_size)]
                map_spec = torch.stack(map_spec, dim=0).reshape((-1, Cs, Hs, Ws))
                if map_bevs_trans is not None:
                    map_bev_trans = torch.stack(map_bev_trans, dim=0).reshape((-1, Ct, Ht, Wt))
                else:
                    map_bev = torch.stack(map_bev, dim=0).reshape((-1, C, H, W))
                    map_bev_trans = trans_cnn(map_bev)
                query_spec_repeated = query_spec.repeat(map_spec.shape[0], 1, 1, 1)

                # Yaw Estimation
                time_start = time.time()
                batch_corrs_yaw, batch_scores_yaw, batch_shifts = estimate_yaw(query_spec_repeated, map_spec)
                batch_angles_rad = batch_shifts * 2 * np.pi / H
                batch_angles_extra_rad = batch_angles_rad - np.pi     

                # Translation Estimation
                if query_bevs_trans is not None:
                    query_bev_trans = query_bevs_trans[query_ndx]
                    query_bev_trans_rotated = rotate_bev_batch(query_bev_trans, to_torch(batch_angles_rad))
                    query_bev_trans_rotated_extra = rotate_bev_batch(query_bev_trans, to_torch(batch_angles_extra_rad))
                else:
                    query_bev_rotated = rotate_bev_batch(query_bev, to_torch(batch_angles_rad))
                    query_bev_rotated_extra = rotate_bev_batch(query_bev, to_torch(batch_angles_extra_rad))                
                    query_bev_trans_rotated = trans_cnn(query_bev_rotated)
                    query_bev_trans_rotated_extra = trans_cnn(query_bev_rotated_extra)             
                x, y, errors, corrs = solve_translation(query_bev_trans_rotated, map_bev_trans, scale=scale)
                x_extra, y_extra, errors_extra, corrs_extra = solve_translation(query_bev_trans_rotated_extra, map_bev_trans, scale=scale)
                batch_angles_rad = batch_angles_rad.detach().cpu().numpy()
                batch_angles_extra_rad = batch_angles_extra_rad.detach().cpu().numpy()
                
                errors = errors.detach().cpu().numpy()
                errors_extra = errors_extra.detach().cpu().numpy()
                batch_errors = [errors[i] if errors[i] < errors_extra[i] else errors_extra[i] for i in range(map_spec.shape[0])]                
                batch_pred_x = [x[i] / (scale * H) * x_range if errors[i] < errors_extra[i] else x_extra[i] / (scale * H) * x_range for i in range(map_spec.shape[0])]  # in meters
                batch_pred_y = [y[i] / (scale * W) * y_range if errors[i] < errors_extra[i] else y_extra[i] / (scale * W) * y_range for i in range(map_spec.shape[0])]  # in meters
                batch_pred_yaw = [batch_angles_rad[i] if errors[i] < errors_extra[i] else batch_angles_extra_rad[i] for i in range(map_spec.shape[0])]  # in radians
                batch_corrs = [corrs[i] if errors[i] < errors_extra[i] else corrs_extra[i] for i in range(map_spec.shape[0])] # correlation map
                batch_corrs = torch.stack(batch_corrs, dim=0)
                batch_pred_x = np.array(batch_pred_x)
                batch_pred_y = np.array(batch_pred_y)
                batch_pred_yaw = np.array(batch_pred_yaw)
                batch_scores_yaw = batch_scores_yaw.detach().cpu().numpy()
                batch_scores_trans = -np.array(batch_errors)
                batch_corrs = batch_corrs.detach().cpu().numpy()
                time_end = time.time()
                # print("Time of batch pose estimation", time_end - time_start)
                if icp_refine:
                    for m in range(len(batch_pred_x)):
                        x, y, yaw = batch_pred_x[m], batch_pred_y[m], batch_pred_yaw[m]
                        T_estimated = xyz_ypr2m(x, y, 0, yaw, 0, 0)
                        _, T_estimated = fast_gicp(query_pc[:,:3], positive_pcs[m][:,:3], init_pose=T_estimated)
                        x, y, _, yaw, _, _ = m2xyz_ypr(T_estimated)
                        batch_pred_x[m], batch_pred_y[m], batch_pred_yaw[m] = x, y, yaw
                
                if i == 0:
                    scores_yaw = batch_scores_yaw
                    scores_trans = batch_scores_trans
                    pred_x = batch_pred_x
                    pred_y = batch_pred_y
                    pred_yaw = batch_pred_yaw
                    corr_likelihood = batch_corrs
                else:
                    scores_yaw = np.concatenate((scores_yaw, batch_scores_yaw), axis=-1)
                    scores_trans = np.concatenate((scores_trans, batch_scores_yaw), axis=-1)
                    pred_x = np.concatenate((pred_x, batch_pred_x), axis=-1)
                    pred_y = np.concatenate((pred_y, batch_pred_y), axis=-1)
                    pred_yaw = np.concatenate((pred_yaw, batch_pred_yaw), axis=-1)
                    corr_likelihood = np.concatenate((corr_likelihood, batch_corrs), axis=-1)
            
            print('Ground truth translation: x: {}, y: {}, rotation: {}'.format(gt_x, gt_y, gt_yaw))
            print('Estimated translation: x: {}, y: {}, rotation: {}'.format(pred_x, pred_y, pred_yaw))

            # Calculate pose errors
            yaw_err = np.abs((gt_yaw - pred_yaw) * 180 / np.pi % 360) # in degrees
            yaw_err = np.minimum(yaw_err, 360 - yaw_err) # circular property
            x_err = np.abs(gt_x - pred_x) # in meters
            y_err = np.abs(gt_y - pred_y) # in meters
            trans_err = np.sqrt(x_err**2 + y_err**2) # in meters
            print(f'Yaw error: {yaw_err}')
            print(f'Translation error: {trans_err}')
            if query_ndx == 0:
                x_errors = x_err
                y_errors = y_err
                trans_errors = trans_err
                yaw_errors = yaw_err
                real_dists = gt_d
            else:
                x_errors = np.concatenate((x_errors, x_err), axis=-1)
                y_errors = np.concatenate((y_errors, y_err), axis=-1)
                trans_errors = np.concatenate((trans_errors, trans_err), axis=-1)
                yaw_errors = np.concatenate((yaw_errors, yaw_err), axis=-1)
                real_dists = np.concatenate((real_dists, gt_d), axis=-1)
            
            yaw_draw = softmax(batch_corrs_yaw)
            
            if do_viz:
                # Successful Case (5deg, 2m)
                if yaw_err[0] < 5 and trans_err[0] < 2:
                    save_folder_path = f'{folder_path}/query_{query_ndx}_{gt_yaw[0]}_{gt_x[0]}_{gt_y[0]}_{gt_d[0]}_{yaw_err[0]}_{x_err[0]}_{y_err[0]}_{trans_err[0]}'
                else:
                    save_folder_path = f'{folder_path}/fail_query_{query_ndx}_{gt_yaw[0]}_{gt_x[0]}_{gt_y[0]}_{gt_d[0]}_{yaw_err[0]}_{x_err[0]}_{y_err[0]}_{trans_err[0]}'
                os.makedirs(save_folder_path, exist_ok=True)
                query_bev_trans = trans_cnn(query_bev.unsqueeze(0))
                corr = batch_corrs[0]
                imshow(query_spec[0].squeeze(), f'{save_folder_path}/query_spec.jpg')
                imshow(map_spec[0][0].squeeze(), f'{save_folder_path}/map_spec.jpg')
                gt_xy_grid = [gt_x_grid[0].detach().cpu().numpy(), gt_y_grid[0].detach().cpu().numpy()]
                est_x_grid = int(H / 2 - np.ceil(pred_x[0] / x_range * H))
                est_y_grid = int(W / 2 - np.ceil(pred_y[0] / y_range * W))
                est_xy_grid = [est_x_grid, est_y_grid]
                if self.params.use_rgb:
                    plot_heatmap(corr, gt_xy_grid, gt_yaw[0], None, pred_yaw[0], f'{save_folder_path}/heatmap.jpg')
                else:
                    plot_heatmap(corr, gt_xy_grid, gt_yaw[0], None, pred_yaw[0], f'{save_folder_path}/heatmap.jpg', sensor='lidar')
                imshow(query_bev_trans[0][0].squeeze(), f'{save_folder_path}/query_bev_trans.jpg')
                imshow(query_bev_trans_rotated[0][0].squeeze(), f'{save_folder_path}/query_bev_trans_rotated.jpg')
                imshow(query_bev_trans_rotated_extra[0][0].squeeze(), f'{save_folder_path}/query_bev_trans_rotated_extra.jpg')
                imshow(map_bev_trans[0][0].squeeze(), f'{save_folder_path}/map_bev_trans.jpg')
                
                # LiDAR Point Clouds
                plot_point_cloud(query_pc, s=0.3, c='g', alpha=0.5, save_path=f'{save_folder_path}/query_pc.jpg')
                plot_point_cloud(positive_pcs[0], s=0.3, c='b', alpha=0.5, save_path=f'{save_folder_path}/map_pc.jpg')
                
                # Camera Images
                if self.params.use_rgb:
                    query_sph_img = generate_sph_image(query_image, self.dataset_type, self.dataset_root)
                    map_sph_img = generate_sph_image(positive_images[0], self.dataset_type, self.dataset_root)
                    if self.dataset_type == 'nclt':
                        query_sph_img = cv2.cvtColor(query_sph_img, cv2.COLOR_RGB2BGR)
                        map_sph_img = cv2.cvtColor(map_sph_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f'{save_folder_path}/query_sph.jpg', query_sph_img)
                    cv2.imwrite(f'{save_folder_path}/map_sph.jpg', map_sph_img)
                    for n in range(query_image.shape[0]):
                        query_image_n = query_image[n]
                        positive_map_image_n = positive_images[0][n]
                        cv2.imwrite(f'{save_folder_path}/query_cam_{n+1}.jpg', query_image_n)
                        cv2.imwrite(f'{save_folder_path}/map_cam_{n+1}.jpg', positive_map_image_n)
        
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
        trans_loss = [t.view(-1) for t in trans_loss]
        trans_loss = torch.cat(trans_loss, dim=0)
        trans_loss = trans_loss.detach().cpu().numpy()
        mean_trans_loss = np.mean(trans_loss)
        np.save(f'{folder_path}/trans_loss.npy', trans_loss)
        plot_yaw_errors(yaw_errors, f'{folder_path}/yaw_errors_distribution.jpg')
        plot_trans_yaw_errors(yaw_errors, trans_errors, f'{folder_path}/trans_yaw_errors.jpg')

        global_metrics = {'dataset_type': self.dataset_type, 'eval_setting': eval_setting, 'positive_threshold': positive_threshold, 'gt_icp_refine': gt_icp_refine, 'estimate_icp_refine': icp_refine, 'total_positives': total_positives, 'mean_trans_loss': mean_trans_loss, 'recall_pe': recall_pe, 'mean_x_error': mean_x_error, 
                          'mean_y_error': mean_y_error, 'mean_trans_error': mean_trans_error, 'mean_yaw_error': mean_yaw_error, 'x_error_quantiles': x_error_quantiles, 'y_error_quantiles': y_error_quantiles, 'trans_error_quantiles': trans_error_quantiles, 'yaw_error_quantiles': yaw_error_quantiles}
        return global_metrics


    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, trans_cnn, phase='map'):
        self.model2eval((model, trans_cnn))
        pcs = []
        images = None
        embeddings = None
        bevs = None
        specs = None
        bevs_trans = None
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
            elif self.params.dataset_type == 'nclt':
                scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
                assert os.path.exists(scan_filepath)
                orig_pc, orig_imgs = self.pcim_loader(scan_filepath, sph)
                bev_path = scan_filepath.replace('velodyne_sync', 'bev')
                bev_path = bev_path.replace('bin', 'npy')
            
            orig_pc = np.array(orig_pc)
            orig_imgs = np.array(orig_imgs)                
            if self.params.use_bev:
                bev = np.load(bev_path)
                bev = torch.from_numpy(bev).float()
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
            
            batch = {'pc': pc, 'img': imgs, 'orig_pc': orig_pc}
            x, bev, spec, bev_trans = self.compute_embedding(batch, model)

            if self.params.use_rgb and images is None:
                images = np.zeros((len(eval_subset), orig_imgs.shape[-4], orig_imgs.shape[-3], orig_imgs.shape[-2], orig_imgs.shape[-1]), dtype=orig_imgs.dtype)           
            if bevs is None:
                bevs = torch.zeros((len(eval_subset), bev.shape[-3], bev.shape[-2], bev.shape[-1])).to(bev.device)                
            if specs is None:
                specs = torch.zeros((len(eval_subset), spec.shape[-3], spec.shape[-2], spec.shape[-1])).to(spec.device)               
            if embeddings is None:
                embeddings = torch.zeros((len(eval_subset), x.shape[-2], x.shape[-1])).to(x.device)   
            
            if bev_trans is not None:
                if bevs_trans is None:
                    bevs_trans = torch.zeros((len(eval_subset), bev_trans.shape[-3], bev_trans.shape[-2], bev_trans.shape[-1])).to(bev_trans.device)  
                bevs_trans[ndx] = bev_trans
            else:
                if phase == 'map':
                    bev_trans = trans_cnn(bev)
                    if bevs_trans is None:
                        bevs_trans = torch.zeros((len(eval_subset), bev_trans.shape[-3], bev_trans.shape[-2], bev_trans.shape[-1])).to(bev_trans.device)  
                    bevs_trans[ndx] = bev_trans
            
            pcs.append(orig_pc[:,:3])
            if self.params.use_rgb:
                images[ndx] = orig_imgs
            bevs[ndx] = bev
            specs[ndx] = spec
            embeddings[ndx] = x

        return pcs, images, bevs, specs, embeddings, bevs_trans


    def compute_embedding(self, batch, model):
        """
        Returns BEV features
        """
        time_start = time.time()

        with torch.no_grad():
            y = model(batch)
            x = y['global']
            bev = y['bev']
            spec = y['spec']
            bev_trans = y['bev_trans']

        time_end = time.time()
        # print('Time of embedding generation: ', time_end-time_start)  

        return x, bev, spec, bev_trans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Fusion model')
    parser.add_argument('--dataset_root', type=str, default='~/Data/NCLT', help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, default='nclt', choices=['mulran', 'southbay', 'kitti', 'nclt', 'oxford'])
    parser.add_argument('--eval_set', type=str, default='test_2012-02-04_2012-03-17_20.0_5.0.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--positive_threshold', type=float, default=10, help='Positive Threshold for Evaluation')        
    parser.add_argument('--radius', type=float, nargs='+', default=[2, 5, 10, 20, 25], help='True positive thresholds in meters')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weight', type=str, default=None, help='Trained global model weight')
    parser.add_argument('--icp_refine', dest='icp_refine', action='store_true')    
    parser.add_argument('--viz', dest='viz', action='store_true', help='Visualize the results')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for evaluation')
    
    args = parser.parse_args()
    dataset_root = os.path.expanduser(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Number of sampled query elements: {args.n_samples}')
    print(f'Model config path: {args.model_config}')
    print(f'Experiment name: {args.exp_name}')
    
    if args.weight is None:
        w = 'RANDOM WEIGHT'
    else:
        w = args.weight
    print(f'Weight: {w}')
    print(f'ICP refine: {args.icp_refine}')
    print(f'Visualization: {args.viz}')
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
    model.print_info()
    
    if model_params.use_rgb:
        feature_dim = backbone_conf['output_channels']
        trans_cnn = last_conv_block(feature_dim, feature_dim, bn=True)
    else:
        if model_params.use_bev:
            feature_dim = model_params.feature_dim
            mid_channels = model_params.feature_dim
        else:
            feature_dim = model_params.feature_dim
            mid_channels = feature_dim
        trans_cnn = last_conv_block(feature_dim, mid_channels, bn=False)
    trans_cnn = trans_cnn.to(device)    
    # corr2soft = (200., 0.)
    # corr2soft = corr2soft.to(device)

    if args.weight is not None:
        assert os.path.exists(args.weight), 'Cannot open network weight: {}'.format(args.weight)
        print('Loading weight: {}'.format(args.weight))
        
        model = nn.DataParallel(model)
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        trans_cnn.load_state_dict(checkpoint['trans_cnn'], strict=False)
        # corr2soft.load_state_dict(checkpoint['corr2soft'], strict=False)

    model.to(device)

    eval_config = {
        'positive_threshold': args.positive_threshold, # estimate relative pose of two images within 10 meters
        'rotation_threshold': 5, # rotation thershold to calculation pose estimation success rate
        'translation_threshold': 2, # translation thershold to calculation pose estimation success rate
    }
    
    with torch.no_grad():
        evaluator = GLEvaluator(dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius,
                                    n_samples=args.n_samples, params=model_params)
        global_metrics = evaluator.evaluate(model, trans_cnn, exp_name=exp_name, icp_refine=args.icp_refine, do_viz=args.viz, config=eval_config)
        global_metrics['weight'] = args.weight
        evaluator.print_results(global_metrics)
        evaluator.export_eval_stats(f'./results/{exp_name}/eval_results_{args.dataset_type}.txt', global_metrics)
