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
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader
from sklearn.neighbors import KDTree
from glnet.models.utils import *
from glnet.config.config import *

from tools.evaluator import Evaluator
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


import struct

def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z


def load_lidar_file_nclt(file_path):
    n_vec = 4
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
        s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)
        
        hits += [[x, y, z]]
        if np.abs(x) < 70. and z > -20. and z < -2. and np.abs(y) < 70. and not(np.abs(x) < 1. and np.abs(y) < 1.):
            hits += [[x, y, z]]
    
    f_bin.close()
    hits = np.asarray(hits)

    return hits


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

    def evaluate(self, model, exp_name=None, normalize=False, ransac=False, config=default_config, *args, **kwargs):
        if exp_name is None:
            exp_name = self.params.model
        if self.dataset_type == 'nclt':
            gt_icp_refine = False # use icp pose refinement
        elif self.dataset_type == 'oxford':
            gt_icp_refine = True # use icp pose refinement

        # map_embeddings = self.compute_embeddings(self.eval_set.map_set, model, normalize)
        # query_embeddings = self.compute_embeddings(self.eval_set.query_set, model, normalize)
        
        map_xys = map_positions = self.eval_set.get_map_positions()
        tree = KDTree(map_positions)
        query_xys = query_positions = self.eval_set.get_query_positions()

        num_maps = len(map_positions)
        num_queries = len(query_positions)
        print(f'{num_maps} database elements, {num_queries} query elements')

        metrics = {'rre': [], 'rte': [], 
                   'success': [], 'rre_refined': [], 
                   'rte_refined': [], 'success_refined': []}
        
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
                query_pc = load_lidar_file_nclt(query_filepath)
            elif self.dataset_type == 'oxford':
                query_filepath = self.eval_set.query_set[query_ndx].filepaths
                query_pc, _ = self.pcim_loader(query_filepath)
            # query_pc, _ = self.pcim_loader(query_filepath)

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
                    map_pc = load_lidar_file_nclt(map_filepath)
                elif self.dataset_type == 'oxford':
                    map_filepath = self.eval_set.map_set[idx].filepaths
                    map_pc, _ = self.pcim_loader(map_filepath)
                # map_pc, _ = self.pcim_loader(map_filepath)
                
                T_gt = relative_pose(query_pose, map_pose)
                # Ground Truth Pose Refinement
                if gt_icp_refine:
                    # _, T_gt, _ = o3d_icp(query_pc[:,:3], map_pc[:,:3], transform=T_gt, point2plane=True)
                    _, T_gt = fast_gicp(query_pc[:,:3], map_pc[:,:3], init_pose=T_gt) # spends less time than open3d
                rel_x, rel_y, rel_z, rel_yaw, rel_pitch, rel_roll = m2xyz_ypr(T_gt)
                gt_d = np.sqrt(rel_x**2 + rel_y**2)
            
                T_estimated = self.compute_transformation(query_pc, map_pc, model, ransac=ransac)
                # try:
                #     T_estimated = self.compute_transformation(query_pc, map_pc, model, ransac=ransac)
                # except:
                #     import pdb; pdb.set_trace()
                
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
                          'ransac': ransac, 'gt_icp_refine': gt_icp_refine, 'estimate_icp_refine': self.icp_refine, 'total_positives': total_positives, 
                          'recall_pe': recall_pe, 'mean_x_error': mean_x_error, 'mean_y_error': mean_y_error, 'mean_trans_error': mean_trans_error, 'mean_yaw_error': mean_yaw_error, 
                          'x_error_quantiles': x_error_quantiles, 'y_error_quantiles': y_error_quantiles, 'trans_error_quantiles': trans_error_quantiles, 'yaw_error_quantiles': yaw_error_quantiles}
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
                pc = load_lidar_file_nclt(scan_filepath)
            elif self.dataset_type == 'oxford':
                scan_filepath = e.filepaths
                assert os.path.exists(scan_filepath[0])
                pc, _ = self.pcim_loader(scan_filepath)
            # pc, _ = self.pcim_loader(scan_filepath)

            global_embedding = self.compute_global_embedding(pc, model, normalize)
            if global_embeddings is None:
                global_embeddings = np.zeros((len(eval_subset), global_embedding.shape[-1]), dtype=global_embedding.dtype)

            global_embeddings[ndx] = global_embedding

        return global_embeddings
    
    
    def compute_global_embedding(self, pc, model, normalize=False, *args, **kwargs):
        model.eval()
        with torch.no_grad():
            pc_list = []
            pc = torch.from_numpy(pc).float().cuda()
            if pc.shape[-1] == 3:
                pc_reflectance = torch.ones(pc.shape[0], 1).to(pc)
                pc = torch.cat((pc, pc_reflectance), 1)
            pc_list.append(model.backbone.prepare_input(pc))
            model_in = KittiDataset.collate_batch(pc_list)
            
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).to(pc)
            del pc
            
            batch_dict = model(model_in, metric_head=False, compute_rotation=False, compute_transl=False)
            
            global_embedding = batch_dict['out_embedding']
            
            if normalize:
                global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)
            
        return global_embedding.detach().cpu().numpy()


    def compute_transformation(self, query_pc, map_pc, model, ransac=False, *args, **kwargs):
        model.eval()
        # import pdb; pdb.set_trace()
        with torch.no_grad():
        # with torch.autocast(device_type='cuda', dtype=torch.half), torch.no_grad():
            pc_list = []
            query_pc = torch.from_numpy(query_pc).cuda().to(torch.float32)
            if query_pc.shape[-1] == 3:
                query_pc_reflectance = torch.ones(query_pc.shape[0], 1).to(query_pc)
                query_pc = torch.cat((query_pc, query_pc_reflectance), 1)
            map_pc = torch.from_numpy(map_pc).to(query_pc)
            if map_pc.shape[-1] == 3:
                map_pc_reflectance = torch.ones(map_pc.shape[0], 1).to(query_pc)
                map_pc = torch.cat((map_pc, map_pc_reflectance), 1)
            
            pc_list.append(model.backbone.prepare_input(query_pc))
            pc_list.append(model.backbone.prepare_input(map_pc))
            model_in = KittiDataset.collate_batch(pc_list)
            
            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().to(query_pc)
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
    # Ignore keypoint saliency and choose keypoint randomly
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for evaluation')
    
    args = parser.parse_args()
    dataset_root = os.path.expanduser(args.dataset_root)
    print(f'Dataset root: {dataset_root}')
    print(f'Dataset type: {args.dataset_type}')
    print(f'Evaluation set: {args.eval_set}')
    print(f'Radius: {args.radius} [m]')
    print(f'Number of sampled query elements: {args.n_samples}')
    print(f'Model config path: {args.model_config}')
    
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

    # model = model_factory(model_params)
    
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
    global_metrics = evaluator.evaluate(model, exp_name=exp_name, normalize=normalize, ransac=args.ransac)
    global_metrics['weight'] = args.weight
    evaluator.print_results(global_metrics)
    folder_path = os.path.expanduser(f'./results/{exp_name}')
    evaluator.export_eval_stats(f'{folder_path}/eval_results_{args.dataset_type}.txt', global_metrics)
