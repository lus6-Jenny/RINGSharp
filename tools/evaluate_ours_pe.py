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
from glnet.models.utils import *
from glnet.config.config import *
from glnet.models.backbones_2d.unet import last_conv_block
from glnet.models.model_factory import model_factory
from glnet.utils.params import TrainingParams, ModelParams
from glnet.utils.data_utils.point_clouds import o3d_icp, fast_gicp
from glnet.utils.data_utils.poses import m2ypr, m2xyz_ypr, relative_pose, relative_pose_batch
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader, get_pointcloud_with_image_loader

from sklearn.neighbors import KDTree
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
from glnet.utils.loss_utils import *
from tools.evaluator import Evaluator
from tools.plot_pose_errors import plot_cdf, plot_yaw_errors, plot_trans_yaw_errors, cal_recall_pe


class GLEvaluator(Evaluator):
    # Evaluation of Pose Estimation
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

    def evaluate(self, model, trans_cnn, exp_name=None, config=default_config):
        if exp_name is None:
            exp_name = self.params.model
        if self.dataset_type == 'nclt':
            icp_refine = False # use icp pose refinement
            pc_bev_conf = nclt_pc_bev_conf
        elif self.dataset_type == 'oxford':
            icp_refine = True # use icp pose refinement
            pc_bev_conf = oxford_pc_bev_conf
        do_argmax = True
                        
        # load embedding model
        self.model2eval((model, trans_cnn))
        query_pcs, query_images, query_bevs, query_specs, query_embeddings, query_bevs_trans = self.compute_embeddings(self.eval_set.query_set, model, trans_cnn, phase='query')
        map_pcs, map_images, map_bevs, map_specs, map_embeddings, map_bevs_trans = self.compute_embeddings(self.eval_set.map_set, model, trans_cnn, phase='map')

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
        batch_size = 2
        if self.params.use_rgb:
            x_range = backbone_conf['x_bound'][1] - backbone_conf['x_bound'][0]
            y_range = backbone_conf['y_bound'][1] - backbone_conf['y_bound'][0]
            ppm = 1 / backbone_conf['x_bound'][2] # pixel per meter
        else:
            x_range = pc_bev_conf['x_bound'][1] - pc_bev_conf['x_bound'][0]
            y_range = pc_bev_conf['y_bound'][1] - pc_bev_conf['y_bound'][0]
            ppm = H / x_range # pixel per meter

        # Pose Estimation Parameters
        total_positives = 0 # total number of positives
        positive_threshold = config['positive_threshold'] # positive threshold for pose estimation
        rotation_threshold = config['rotation_threshold'] # rotation threshold to calculation pose estimation success rate
        translation_threshold = config['translation_threshold'] # translation thershold to calculation pose estimation success rate  
        quantiles = [0.25, 0.5, 0.75, 0.95]
        template_sampler = TemplateSampler(H, optimize=False)
        kl_map = init_kl_map(H, 3)
        
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
            if icp_refine:
                rel_poses_refine = []
                for i, rel_pose in enumerate(rel_poses):
                    # _, rel_pose, _ = o3d_icp(query_pc[:,:3], positive_pcs[i][:,:3], transform=rel_pose.detach().cpu().numpy(), point2plane=True)   
                    _, rel_pose = fast_gicp(query_pc[:,:3], positive_pcs[i][:,:3], init_pose=rel_pose.detach().cpu().numpy()) # spends less time than open3d  
                rel_poses_refine.append(rel_pose)       
                rel_poses = array2tensor(np.array(rel_poses_refine))            
            raw_x, raw_y, raw_z, raw_yaw, raw_pitch, raw_roll = m2xyz_ypr(rel_poses)
            if self.params.use_rgb:
                gt_yaw = -raw_yaw * 180 / np.pi # in degrees
                gt_x = raw_y # in meters
                gt_y = raw_x # in meters                             
            else:
                gt_yaw = raw_yaw * 180 / np.pi # in degrees
                gt_x = raw_x # in meters
                gt_y = raw_y # in meters
            gt_yaw_grid = np.ceil(gt_yaw * H / 360.)
            gt_d = np.sqrt(gt_x**2 + gt_y**2) # in meters

            if query_bevs_trans is not None:
                query_bev_trans = query_bevs_trans[query_ndx]
                query_bev_trans_repeated = query_bev_trans.repeat(batch_size, 1, 1, 1)
                query_bev_trans = template_sampler(query_bev_trans_repeated).reshape(batch_size, -1, Ct, Ht, Wt)
            else:
                query_bev_repeated = query_bev.repeat(batch_size, 1, 1, 1)
                query_bev_templates = template_sampler(query_bev_repeated).reshape(-1, C, H, W)
                query_bev_trans = trans_cnn(query_bev_templates).reshape(batch_size, -1, Ct, Ht, Wt)

            # ------------ Pose Estimation ------------
            if num_positives % batch_size == 0:
                batch_num = num_positives // batch_size
            else:
                batch_num = num_positives // batch_size + 1      
            for i in range(batch_num):
                if i == batch_num - 1:
                    if map_bevs_trans is not None:   
                        map_bev_trans = [positive_bevs_trans[k] for k in range(i*batch_size, num_positives)]
                    else:
                        map_bev = [positive_bevs[k] for k in range(i*batch_size, num_positives)]
                    num = len(range(i*batch_size, num_positives))
                    if query_bevs_trans is not None:
                        query_bev_trans = query_bevs_trans[query_ndx]
                        query_bev_trans_repeated = query_bev_trans.repeat(num, 1, 1, 1)
                        query_bev_trans = template_sampler(query_bev_trans_repeated).reshape(num, -1, Ct, Ht, Wt)
                    else:
                        query_bev_repeated = query_bev.repeat(num, 1, 1, 1)
                        query_bev_templates = template_sampler(query_bev_repeated).reshape(-1, C, H, W)
                        query_bev_trans = trans_cnn(query_bev_templates).reshape(num, -1, Ct, Ht, Wt)                        
                else:
                    if map_bevs_trans is not None:
                        map_bev_trans = [positive_bevs_trans[k] for k in range(i*batch_size, (i+1)*batch_size)]
                    else:
                        map_bev = [positive_bevs[k] for k in range(i*batch_size, (i+1)*batch_size)]
                if map_bevs_trans is not None:
                    map_bev_trans = torch.stack(map_bev_trans, dim=0).reshape((-1, Ct, Ht, Wt))
                else:
                    map_bev = torch.stack(map_bev, dim=0).reshape((-1, C, H, W))
                    map_bev_trans = trans_cnn(map_bev)

                scores = conv2d_fft_batchwise(
                    map_bev_trans.float(),
                    query_bev_trans.float(),
                )
                scores = scores.moveaxis(1, -1)  # B,H,W,N
                scores = torch.fft.fftshift(scores, dim=(-3, -2))
                log_probs = log_softmax_spatial(scores)
                with torch.no_grad():
                    uvr_max = argmax_xyr(scores).to(scores)
                    uvr_avg, _ = expectation_xyr(log_probs.exp())

                if do_argmax:
                    # Argmax
                    x = uvr_max[..., 0].detach().cpu().numpy() / ppm
                    y = uvr_max[..., 1].detach().cpu().numpy() / ppm
                    yaw = uvr_max[..., 2].detach().cpu().numpy()
                else:
                    # Average
                    x = uvr_avg[..., 0].detach().cpu().numpy() / ppm
                    y = uvr_avg[..., 1].detach().cpu().numpy() / ppm
                    yaw = uvr_avg[..., 2].detach().cpu().numpy()

                yaw_draw = torch.sum(log_probs.exp(), dim=(-3, -2))

                if i == 0:
                    pred_x = x
                    pred_y = y
                    pred_yaw = yaw
                else:
                    pred_x = np.concatenate((pred_x, x), axis=-1)
                    pred_y = np.concatenate((pred_y, y), axis=-1)
                    pred_yaw = np.concatenate((pred_yaw, yaw), axis=-1)     
            
            print('Ground truth translation: x: {}, y: {}, rotation: {}'.format(gt_x, gt_y, gt_yaw))
            print('Estimated translation: x: {}, y: {}, rotation: {}'.format(pred_x, pred_y, pred_yaw))

            # Calculate pose errors
            yaw_err = np.abs(pred_yaw % 360 - gt_yaw % 360) # in degrees
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

            # Visualization
            if query_ndx % 10 == 0:
                # self.save_image(f'{folder_path}/query_bev_trans_{query_ndx}.jpg', query_bev_trans[0].squeeze())
                # self.save_image(f'{folder_path}/query_bev_trans_extra_{query_ndx}.jpg', query_bev_trans_extra[0].squeeze())
                # self.save_image(f'{folder_path}/map_bev_trans_{query_ndx}.jpg', map_bev_trans[0].squeeze())
                plot_range = np.arange(0, H)
                save_path = f'{folder_path}/yaw_distribution_{query_ndx}.jpg'
                gt_kl_map = F.normalize(kl_map[torch.from_numpy(gt_yaw_grid)[0].long()], p=1, dim=0)
                visualize_dist(plot_range, yaw_draw[0].detach().cpu().numpy().squeeze(), gt_kl_map.detach().cpu().numpy().squeeze(), save_path)

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
        plot_yaw_errors(yaw_errors, f'{folder_path}/yaw_errors_distribution.jpg')
        plot_trans_yaw_errors(yaw_errors, trans_errors, f'{folder_path}/trans_yaw_errors.jpg')
        
        global_metrics = {'dataset_type': self.dataset_type, 'eval_setting': eval_setting, 'positive_threshold': positive_threshold, 'icp_refine': icp_refine, 'do_argmax': do_argmax, 'total_positives': total_positives, 'recall_pe': recall_pe, 'mean_x_error': mean_x_error, 
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

            x, bev, spec, bev_trans = self.compute_embedding(pc, imgs, model) 

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
    

    def compute_embedding(self, pc, imgs, model):
        '''
        Returns BEV features
        '''
        time_start = time.time()

        with torch.no_grad():
            batch = {'pc': pc, 'img': imgs}
            y = model(batch)
            x = y['global']
            bev = y['bev']
            spec = y['spec']
            bev_trans = y['bev_trans']            

        time_end = time.time()
        # print('Time of embedding generation: ', time_end-time_start)  

        return x, bev, spec, bev_trans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Fusion model')
    parser.add_argument('--dataset_root', type=str, default='~/Data/NCLT', help='Path to the dataset root')
    parser.add_argument('--dataset_type', type=str, default='nclt', choices=['mulran', 'southbay', 'kitti', 'nclt', 'oxford'])
    parser.add_argument('--eval_set', type=str, default='test_2012-02-04_2012-03-17_20.0_5.0.pickle', help='File name of the evaluation pickle (must be located in dataset_root')
    parser.add_argument('--radius', type=float, nargs='+', default=[2, 5, 10, 20, 25], help='True positive thresholds in meters')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of elements sampled from the query sequence')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the global model configuration file')
    parser.add_argument('--weight', type=str, default=None, help='Trained global model weight')
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
    print('')

    model_params = ModelParams(args.model_config, args.dataset_type, dataset_root)
    model_params.print()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Device: {}'.format(device))

    if args.exp_name is None:
        exp_name = model_params.model
    else:
        exp_name = args.exp_name
    
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
        
        model = nn.DataParallel(model)
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        trans_cnn.load_state_dict(checkpoint['trans_cnn'], strict=False)

    model.to(device)
    
    with torch.no_grad():
        evaluator = GLEvaluator(dataset_root, args.dataset_type, args.eval_set, device, radius=args.radius,
                                    n_samples=args.n_samples, params=model_params)
        global_metrics = evaluator.evaluate(model, trans_cnn, exp_name=exp_name)
        global_metrics['weight'] = args.weight
        evaluator.print_results(global_metrics)
        evaluator.export_eval_stats(f'./results/{exp_name}/eval_results_{args.dataset_type}.txt', global_metrics)
