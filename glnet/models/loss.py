
# Zhejiang University
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance
from glnet.utils.params import ModelParams
from glnet.utils.loss_utils import *
from glnet.utils.common_utils import _ex, to_numpy, to_torch
from glnet.models.utils import *
import glnet.utils.vox_utils.geom as geom
import glnet.utils.vox_utils.basic as basic
from glnet.utils.data_utils.point_clouds import o3d_icp, fast_gicp
from glnet.utils.data_utils.poses import apply_transform, m2ypr, m2xyz_ypr
from glnet.models.localizer.circorr2 import circorr2
from torch.cuda.amp.autocast_mode import autocast
from glnet.config.config import *
from glnet.models.voting import (
    make_grid,
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xy,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
    TemplateSampler,
)


def make_losses(params: ModelParams):
    if params.loss == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        pr_loss_fn = BatchHardTripletLossWithMasks(params.margin)
    elif params.loss == 'BatchHardContrastiveLoss':
        pr_loss_fn = BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin)
    elif params.loss == 'PRLoss':
        pr_loss_fn = PRLoss(params)
    else:
        pr_loss_fn = None
        print('Unknown loss: {}'.format(params.loss))
    
    if 'disco' in params.model:
        yaw_loss_fn = YawLossWithFFT(params)
    else:
        yaw_loss_fn = YawLoss(params)
    
    trans_loss_fn = TransLoss(params)
    gl_loss_fn = GLLoss(params)
    depth_loss_fn = DepthLoss(params)
    
    return pr_loss_fn, yaw_loss_fn, trans_loss_fn, gl_loss_fn, depth_loss_fn


def save_image(image, save_path):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy() 
    if image.ndim > 2:
        image = image.squeeze()
    image = (image-np.min(image))/(np.max(image)-np.min(image))*255
    image = image.astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, image)


def to_torch(x):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    return x


class DepthLoss:
    def __init__(self, params):
        self.params = params
        self.data_return_depth = True
        self.downsample_factor = backbone_conf['downsample_factor']
        self.dbound = backbone_conf['d_bound']
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
            
    def __call__(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels.to(depth_preds.device)
        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))
        
        loss = 3.0 * depth_loss
        stats = {'depth_loss': loss.item()}

        return loss, stats

    def get_downsampled_gt_depth(self, gt_depths):
        '''
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        '''
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()


class YawLoss:
    def __init__(self, params):
        self.params = params        
        self.softmax = nn.Softmax(dim=-1)
        self.yaw_klloss = nn.KLDivLoss(reduction='sum')
        self.yaw_celoss = nn.CrossEntropyLoss(reduction='sum')

    def __call__(self, specs, poses, positives_mask, kl_map):
        B, C, H, W = specs.shape
        dummy_labels = torch.arange(B)
        loss = torch.zeros(1).cuda()
        circorr = circorr2(is_circular=True, zero_mean_normalize=True)

        for ndx, mask in enumerate(positives_mask):
            query_spec = specs[ndx]
            query_pose = poses[ndx]
            pos_idxs = dummy_labels[positives_mask[ndx]].long()
            num_positives = len(pos_idxs)
            pos_specs = specs[pos_idxs]
            pos_poses = poses[pos_idxs]
            
            query_pose_repeated = query_pose.repeat(num_positives, 1, 1)
            query_spec_repeated = query_spec.repeat(num_positives, 1, 1, 1)

            # GT Relative Pose
            rel_poses = torch.matmul(pos_poses.inverse(), query_pose_repeated)
            raw_x, raw_y, raw_z, raw_yaw, raw_pitch, raw_roll = m2xyz_ypr(rel_poses)
            if self.params.use_rgb:
                gt_yaw = to_torch(-raw_yaw) # in radians
            else:
                gt_yaw = to_torch(raw_yaw) # in radians
            
            # Yaw Correlation
            corrs, scores, shifts = circorr(query_spec_repeated, pos_specs)
            ang_res = corrs.shape[-1]

            # KL Loss
            gt_yaw_grid = ang_res / 2 - torch.ceil(gt_yaw * ang_res / (2 * torch.pi))

            gt_kl_map = torch.zeros((num_positives, ang_res)).cuda()
            for i in range(gt_yaw_grid.shape[0]):
                gt_yaw_grid_i = gt_yaw_grid[i]
                if gt_yaw_grid_i < ang_res / 2:
                    gt_yaw_grid_extra_i = gt_yaw_grid_i + ang_res / 2
                else:
                    gt_yaw_grid_extra_i = gt_yaw_grid_i - ang_res / 2
                gt_kl_map_i = kl_map[gt_yaw_grid_i.long()] + kl_map[gt_yaw_grid_extra_i.long()]
                gt_kl_map_i = F.normalize(gt_kl_map_i, p=1, dim=0)
                gt_kl_map[i] = gt_kl_map_i
            
            pred_kl_map = F.log_softmax(corrs, dim=-1)
            yaw_loss = self.yaw_klloss(pred_kl_map, gt_kl_map)
            # print('yaw_loss:', yaw_loss)
            loss += yaw_loss.to(loss.device)

        stats = {'yaw_loss': loss.item()}

        return loss, stats


class TransLoss:
    def __init__(self, params):
        self.params = params
        if params.use_rgb:
            self.x_range = backbone_conf['x_bound'][1] - backbone_conf['x_bound'][0]
            self.y_range = backbone_conf['y_bound'][1] - backbone_conf['y_bound'][0]
        else:
            self.x_range = nclt_pc_bev_conf['x_bound'][1] - nclt_pc_bev_conf['x_bound'][0]
            self.y_range = nclt_pc_bev_conf['y_bound'][1] - nclt_pc_bev_conf['y_bound'][0]         
        self.softmax = nn.Softmax(dim=-1)
        self.x_loss = nn.CrossEntropyLoss(reduction='sum')
        self.y_loss = nn.CrossEntropyLoss(reduction='sum')
        self.trans_Ploss = Max_P_2DLoss()
        self.trans_Eloss = Exp_2DLoss()

    def __call__(self, bevs, poses, positives_mask, trans_cnn=None):
        B, C, H, W = bevs.shape
        dummy_labels = torch.arange(B)
        loss = torch.zeros(1).cuda()
        
        for ndx, mask in enumerate(positives_mask):
            query_bev = bevs[ndx]
            query_pose = poses[ndx]
            pos_idxs = dummy_labels[positives_mask[ndx]].long()
            num_positives = len(pos_idxs)
            pos_bevs = bevs[pos_idxs]
            pos_poses = poses[pos_idxs]
            
            # GT Relative Pose
            query_pose_repeated = query_pose.repeat(num_positives, 1, 1)
            rel_poses = torch.matmul(pos_poses.inverse(), query_pose_repeated)
            raw_x, raw_y, raw_z, raw_yaw, raw_pitch, raw_roll = m2xyz_ypr(rel_poses)
            if self.params.use_rgb:
                gt_yaw = to_torch(-raw_yaw) # in radians
                gt_x = to_torch(raw_y) # in meters
                gt_y = to_torch(raw_x) # in meters                          
            else:
                gt_yaw = to_torch(raw_yaw) # in radians
                gt_x = to_torch(raw_x) # in meters
                gt_y = to_torch(raw_y) # in meters            

            gt_x_grid = (H / 2 - torch.ceil(gt_x * H / self.x_range)).cuda() # in grids
            gt_y_grid = (W / 2 - torch.ceil(gt_y * W / self.y_range)).cuda() # in grids

            query_bev_rotated = rotate_bev_batch(query_bev, gt_yaw.to(query_bev))
            if trans_cnn is not None:
                query_bev_trans = trans_cnn(query_bev_rotated)
                pos_bevs_trans = trans_cnn(pos_bevs)
            else:
                query_bev_trans = query_bev_rotated
                pos_bevs_trans = pos_bevs
                
            # Translation Correlation
            pred_x, pred_y, errors, corrs_trans = solve_translation(query_bev_trans, pos_bevs_trans)

            # # Seperate X Loss & Y Loss
            # x_loss = self.x_loss(corrs_x, gt_x_grid.long())
            # y_loss = self.y_loss(corrs_y, gt_y_grid.long())
            # trans_loss = x_loss + y_loss
            # loss += trans_loss.to(loss.device)
            # NLL Loss
            nll_loss = self.trans_Ploss(corrs_trans, gt_x_grid, gt_y_grid).sum()
            loss += nll_loss.to(loss.device)
            # Expectation Loss
            # exp_loss = self.trans_Eloss(corrs_trans, gt_x_grid, gt_y_grid)
            # print('NLL Loss:', nll_loss)
            # print('Expectation Loss:', exp_loss)
            # loss += 0.5 * nll_loss.to(loss.device) + 0.5 * exp_loss.to(loss.device)
        
        stats = {'trans_loss': loss.item()}
        
        return loss, stats


class YawLossWithFFT:
    def __init__(self, params):
        self.params = params
        self.celoss = nn.CrossEntropyLoss(reduction='sum')
        self.klloss = nn.KLDivLoss(reduction='sum')
    
    def __call__(self, bevs, poses, positives_mask, kl_map=None):
        B, C, H, W = bevs.shape
        dummy_labels = torch.arange(bevs.shape[0])
        loss = torch.zeros(1).cuda()
        
        for ndx, mask in enumerate(positives_mask):
            query_bev = bevs[ndx]
            query_pose = poses[ndx]
            pos_idxs = dummy_labels[positives_mask[ndx]].long()
            num_positives = len(pos_idxs)
            pos_bevs = bevs[pos_idxs]
            pos_poses = poses[pos_idxs]
            query_bev_repeated = query_bev.repeat(num_positives, 1, 1, 1)
            
            # GT Relative Pose
            query_pose_repeated = query_pose.repeat(num_positives, 1, 1)
            rel_poses = torch.matmul(pos_poses.inverse(), query_pose_repeated)
            raw_x, raw_y, raw_z, raw_yaw, raw_pitch, raw_roll = m2xyz_ypr(rel_poses)
            pred_yaw, corr = phase_corr(query_bev_repeated, pos_bevs)
            
            soft = nn.Softmax(dim=-1)
            
            if self.params.use_rgb:
                gt_yaw = to_torch(-raw_yaw) # in radians
            else:
                gt_yaw = to_torch(raw_yaw) # in radians
            
            gt_yaw_grid = (torch.ceil(gt_yaw / torch.pi * W / 2) + W / 2 - 1).cuda()
            gt_kl_map = torch.zeros((num_positives, W)).cuda()
            
            ### KL Loss ###
            if kl_map is not None:
                for i in range(num_positives):
                    gt_yaw_grid_i = gt_yaw_grid[i]
                    gt_kl_map_i = kl_map[gt_yaw_grid_i.long()]
                    gt_kl_map_i = F.normalize(gt_kl_map_i, p=1, dim=0)
                    gt_kl_map[i] = gt_kl_map_i
            #     corr_in = F.log_softmax(corr, dim=-1)
            #     corr_in = corr_in.reshape(1, -1)
            #     yaw_loss = self.klloss(corr_in, gt_kl_map_i.unsqueeze(0))
            # else:
            #     yaw_loss = self.celoss(corr, gt_yaw_grid.long())
            
            yaw_loss = self.celoss(corr, gt_yaw_grid.long())
            loss += 0.01 * yaw_loss.to(loss.device)
        
        stats = {'yaw_loss': loss.item()}
        
        return loss, stats


class Max_P_2DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, corr, gt_x, gt_y):
        log_corr = F.log_softmax(corr.flatten(-2), dim=-1).reshape(corr.shape)
        value = torch.zeros_like(gt_x)
        for i in range(gt_x.shape[0]):
            value[i] = log_corr[i, gt_x[i].long(), gt_y[i].long()]
        loss = -value.reshape(-1)
        return loss


class Exp_2DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='sum')
        
    def forward(self, corr, gt_x, gt_y):
        h, w = corr.shape[-2:]
        log_corr = F.log_softmax(corr.flatten(-2), dim=-1).reshape(corr.shape)
        exp_xy = expectation_xy(log_corr)
        exp_x = exp_xy[0] * (backbone_conf['x_bound'][1] - backbone_conf['x_bound'][0]) / h
        exp_y = exp_xy[1] * (backbone_conf['y_bound'][1] - backbone_conf['y_bound'][0]) / w
        loss = self.loss_fn(exp_x, gt_x) + self.loss_fn(exp_y, gt_y)
        return loss    


class BEVOccupancyLoss:
    def __init__(self, params):
        self.params = params

    def __call__(self, bevs, pc_bevs):
        # loss_fn = nn.L1Loss(reduce=True, size_average=False)
        loss_fn = nn.MSELoss(reduction='sum')
        loss = loss_fn(bevs, pc_bevs)
        stats = {'occ_loss': loss.item()}

        return loss, stats
    

class PRLoss:
    # place recognition loss
    def __init__(self, params):
        self.params = params
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.circorr = circorr2(is_circular=True, zero_mean_normalize=True)
    
    def __call__(self, specs, positives_mask, negatives_mask):
        '''
        specs: spectrum generated by RT and FFT on yaw BEV feats (B, C, H, W)
        positives_mask: positive masks
        negatives_mask: negative masks
        '''
        dummy_labels = torch.arange(specs.shape[0]).to(specs.device)
        loss = torch.zeros(1).cuda()
        
        for ndx, mask in enumerate(positives_mask):
            # find positive masks
            query_spec = specs[ndx]
            
            pos_idxs = dummy_labels[positives_mask[ndx]]
            neg_idxs = dummy_labels[negatives_mask[ndx]]
            # print('pos_idxs', dummy_labels[positives_mask[ndx]], pos_idxs)
            total_idxs = torch.cat((pos_idxs, neg_idxs), dim=0)
            total_specs = specs[total_idxs]

            query_specs = query_spec.repeat(total_specs.shape[0], 1, 1, 1)
            corrs, scores, angles = self.circorr(query_specs, total_specs)
            
            # Cross Entropy Loss
            for j in range(len(pos_idxs)):
                pred = torch.cat((scores[j].unsqueeze(0), scores[len(pos_idxs):]), dim=0)
                # print('total pred: ', torch.argmax(pred))
                target = torch.tensor(0).long().cuda()
                loss += self.loss_fn(pred, target) 

        stats = {'pr_loss': loss.item()}

        return loss, stats


class GLLoss:
    # global localization loss
    def __init__(self, params):
        self.params = params
        self.pr_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.yaw_loss_fn = nn.KLDivLoss(reduction='sum')
        self.trans_loss_fn = Max_P_2DLoss()
        self.circorr = circorr2(is_circular=True, zero_mean_normalize=False)
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, specs, bevs, poses, corr2soft, positives_mask, negatives_mask, kl_map):
        '''
        specs: spectrum generated by RT and FFT on yaw BEV feats (B, C, H, W)
        bevs: translation BEV feats (B, C, H, W)
        poses: ground truth poses (B, 4, 4)
        corr2soft: softmax function
        positives_mask: positive masks
        negatives_mask: negative masks
        kl_map: KL map for yaw loss
        '''
        __p = lambda x: basic.pack_seqdim(x, B)
        __u = lambda x: basic.unpack_seqdim(x, B)

        dummy_labels = torch.arange(specs.shape[0])
        loss = torch.zeros(1).cuda()
        pr_loss = torch.zeros(1).cuda()
        yaw_loss = torch.zeros(1).cuda()
        trans_loss = torch.zeros(1).cuda()
        
        B, C, H, W = bevs.shape

        for ndx, mask in enumerate(positives_mask):
            # find positive masks
            qspec = specs[ndx]
            qbev = bevs[ndx]
            qpose = poses[ndx]

            # pos_idx = dummy_labels[positives_mask[ndx]][0].item()
            pos_idxs = dummy_labels[positives_mask[ndx]]
            neg_idxs = dummy_labels[negatives_mask[ndx]]
            total_idxs = torch.cat((pos_idxs, neg_idxs), dim=0)
            total_specs = specs[total_idxs]
            total_bevs = bevs[total_idxs]
            total_poses = poses[total_idxs]

            qspecs = qspec.repeat(total_specs.shape[0], 1, 1, 1)
            corrs, scores, angles = self.circorr(qspecs, total_specs)
            ang_res = corrs.shape[-1]
            pred_yaws = angles * 2 * torch.pi / ang_res

            # ground truth poses
            trans_scores = torch.zeros(total_specs.shape[0]).to(scores.device)
            for i, mpose in enumerate(total_poses):
                rel_pose = torch.linalg.inv(mpose) @ qpose
                raw_x, raw_y, raw_z, raw_yaw, raw_pitch, raw_roll = m2xyz_ypr(rel_pose)
                rel_yaw = -raw_yaw # bev depth
                rel_x = raw_y # bev depth
                rel_y = raw_x # bev depth

                qbev_rotated = rotate_bev(qbev, rel_yaw)
                bev = total_bevs[i]
                x, y, error, corr_trans = solve_translation(qbev_rotated, bev, zero_mean_normalize=False)  
                trans_scores[i] = -error
                
                if i <= len(pos_idxs) - 1:
                    # ------ Yaw Loss ------ 
                    corr = corrs[i]
                    gt_yaw = (ang_res / 2) - torch.ceil(rel_yaw / torch.pi * ang_res / 2.)
                    
                    if (gt_yaw < ang_res / 2):
                        gt_yaw2 = gt_yaw + ang_res / 2
                    else:
                        gt_yaw2 = gt_yaw - ang_res / 2
        
                    gt_dist = kl_map[gt_yaw.long()] + kl_map[gt_yaw2.long()]
                    gt_dist = F.normalize(gt_dist, p=1, dim=0)
                    
                    corr_in = F.log_softmax(corr, dim=-1)
                    yaw_loss += self.yaw_loss_fn(corr_in.reshape(1,-1), gt_dist.unsqueeze(0))
                    
                    # ------ Translation Loss ------
                    corr_trans = (corr_trans.squeeze()).cuda() # squeeze the channel
                    gt_x = (H / 2) - torch.ceil((rel_x) / backbone_conf['x_bound'][1] * H / 2.) 
                    gt_x = gt_x.cuda()

                    gt_y = (W / 2) - torch.ceil((rel_y) / backbone_conf['y_bound'][1] * W / 2.) 
                    gt_y = gt_y.cuda()
                    trans_loss += self.trans_loss_fn(corr_trans, gt_x, gt_y)
            
            # ------ PR Loss ------
            for j in range(len(pos_idxs)):
                # print('scores: ', scores, trans_scores)
                pred = torch.cat((scores[j].unsqueeze(0), scores[len(pos_idxs):]), dim=0)
                pred = corr2soft(pred) + torch.cat((trans_scores[j].unsqueeze(0), trans_scores[len(pos_idxs):]), dim=0)
                # print('total pred: ', torch.argmax(pred))
                target = torch.tensor(0).long().cuda()
                pr_loss += self.pr_loss_fn(pred, target)

        stats = {'yaw_loss': yaw_loss.item(), 'trans_loss': trans_loss.item(), 'pr_loss': pr_loss.item()}

        return yaw_loss, trans_loss, pr_loss, stats                


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=True, distance=self.distance,
                                                reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = 5 * self.loss_fn(embeddings, dummy_labels, hard_triplets)

        stats = {'pr_loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }
        
        return loss, stats, hard_triplets


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin, neg_margin):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use contrastive loss with squared Euclidean distance
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance, reducer=reducer_fn, collect_stats=True)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'pr_loss': loss.item(), 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_threshold': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_threshold,
                 'neg_pairs_above_threshold': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_threshold,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss.item(),
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss.item(),
                 'num_pairs': 2*len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets


class ExhausitiveVotingLoss:
    def __init__(self, params, temperature=None):
        self.params = params
        self.temperature = temperature
        self.ppm = 1 / backbone_conf['x_bound'][2] # pixel per meter
        self.num_rotations = 128
        self.template_sampler = TemplateSampler(self.num_rotations)
        self.l1_loss = nn.L1Loss(reduction='sum')

    def __call__(self, bevs, poses, positives_mask):
        dummy_labels = torch.arange(bevs.shape[0])
        loss = torch.zeros(1).cuda()
        B, C, H, W = bevs.shape
        
        query_bevs = torch.zeros(B, C, H, W).to(bevs.device)
        pos_bevs = torch.zeros(B, C, H, W).to(bevs.device)
        gt_xyr = torch.zeros(B, 3).to(bevs.device)
        for ndx, mask in enumerate(positives_mask):
            qpose = poses[ndx]
            qbev = bevs[ndx]
            query_bevs[ndx] = qbev

            pos_idx = dummy_labels[positives_mask[ndx]][0].item()
            pos_pose = poses[pos_idx]
            pos_bev = bevs[pos_idx]
            pos_bevs[ndx] = pos_bev

            # gt_poses
            rel_pose = torch.linalg.inv(pos_pose) @ qpose
            raw_x, raw_y, raw_z, raw_yaw, raw_pitch, raw_roll = m2xyz_ypr(rel_pose)

            rel_yaw = (-raw_yaw * 180 / torch.pi) % 360
            rel_x = raw_y * self.ppm
            rel_y = raw_x * self.ppm

            gt_xyr[ndx, 0] = rel_x
            gt_xyr[ndx, 1] = rel_y
            gt_xyr[ndx, 2] = rel_yaw

        # template matching to estimate 3dof pose between qbev and pos_bev
        scores, log_probs = self.exhaustive_voting(query_bevs, pos_bevs)

        # classification loss
        # nll_loss = nll_loss_xyr(log_probs, gt_xyr)
        
        # regression loss
        uvr_avg, _ = expectation_xyr(log_probs.exp())
        reg_loss = self.l1_loss(uvr_avg, gt_xyr)
        
        # print('nll_loss: ', nll_loss.sum())
        # print('reg_loss: ', reg_loss)
        # loss += nll_loss.sum().to(loss.device)
        loss += reg_loss.to(loss.device)

        stats = {'3dof_loss': loss.item()}
        
        return loss, stats

            
    def exhaustive_voting(self, f_bev, f_map, confidence_bev=None):
        if self.params.model_params.normalize:
            f_bev = F.normalize(f_bev, dim=1)
            f_map = F.normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)

        templates = self.template_sampler(f_bev)
        with torch.cuda.amp.autocast(enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float()
            )
        if self.temperature is not None:
            scores = scores * torch.exp(self.temperature)

        scores = scores.moveaxis(1, -1)  # B,H,W,N
        scores = torch.fft.fftshift(scores, dim=(-3, -2))
        log_probs = log_softmax_spatial(scores)

        return scores, log_probs
