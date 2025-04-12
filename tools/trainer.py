# Zhejiang University

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import time
import tqdm
import numpy as np
import torch
import torch.nn as nn
from glnet.utils.params import TrainingParams, get_datetime
from glnet.models.loss import make_losses
from glnet.models.backbones_2d.unet import last_conv_block
from glnet.models.model_factory import model_factory
from glnet.datasets.dataset_utils import make_dataloaders
from glnet.utils.loss_utils import *
from glnet.config.config import *
from glnet.utils.common_utils import _ex
from tensorboardX import SummaryWriter

random_seed = 123
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)

def print_stats(stats, phase):
    if 'num_triplets' in stats:
        # For triplet loss
        s = '{} - PR loss: {:.6f}   Embedding norm: {:.4f}   Triplets (all): {:.1f}'
        print(s.format(phase, stats['pr_loss'], stats['avg_embedding_norm'], stats['num_triplets']))
    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    
    if 'pr_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'PR loss: {:.6f} '
        l += [stats['pr_loss']]

    if 'yaw_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Yaw loss: {:.6f} '
        l += [stats['yaw_loss']]

    if 'trans_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Trans loss: {:.6f} '
        l += [stats['trans_loss']]

    if 'depth_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Depth loss: {:.6f} '
        l += [stats['depth_loss']]
    
    if len(l) > 0:
        print(s.format(*l))


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def do_train(params: TrainingParams, exp_name, resume=False, debug=False, weight=None, do_val=False, device='cpu'):
    # Import evaluator
    if 'ring_sharp' in params.model_params.model:
        # from evaluate_ours_pe import GLEvaluator # PE
        from evaluate_ours_gl import GLEvaluator # GL
    else:
        from evaluate_others_gl import GLEvaluator
    
    # Create model class
    s = get_datetime()
    model = model_factory(params.model_params)
    model_name = 'model_' + params.model_params.model + '_' + s
    print('Model name: {}'.format(model_name))
    weights_path = _ex(f'./results/weights/{exp_name}')
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    train_log_path = _ex(f'./results/tensorboard/{exp_name}')
    train_writer = SummaryWriter(train_log_path)

    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))
    
    # Move the model to the proper device before configuring the optimizer
    model.to(device)
    model = nn.DataParallel(model)
    if params.model_params.use_rgb:
        feature_dim = backbone_conf['output_channels']
        trans_cnn = last_conv_block(feature_dim, feature_dim, bn=True) 
    else:
        if params.model_params.use_bev:
            feature_dim = params.model_params.feature_dim
            mid_channels = params.model_params.feature_dim
        else:
            feature_dim = params.model_params.feature_dim
            mid_channels = feature_dim
        trans_cnn = last_conv_block(feature_dim, mid_channels, bn=False)
    trans_cnn = trans_cnn.to(device)
    
    if resume:
        if 'netvlad_pretrain' in params.model_params.model:
            checkpoint = torch.load(weight, map_location=lambda storage, loc: storage)
            for key in list(checkpoint['state_dict'].keys()):
                new_key = 'module.' + key
                print('Renaming {} to {}'.format(key, new_key))
                checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)
            model.load_state_dict(checkpoint['state_dict'])
        elif 'ring_sharp' in params.model_params.model:
            checkpoint = torch.load(weight, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=False)
            trans_cnn.load_state_dict(checkpoint['trans_cnn'], strict=True)
        else:
            checkpoint = torch.load(weight, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=True)
        if 'netvlad_pretrain' in params.model_params.model:
            start_epoch = 1
            start_iter = 1
        else:
            start_epoch = checkpoint.get('epoch', 1) + 1
            start_iter = checkpoint.get('iter', 1) + 1
        print('Resume from epoch: {}, iter: {}'.format(start_epoch, start_iter))
    else:
        start_epoch = 1
        start_iter = 1

    # Set up dataloaders
    dataloaders = make_dataloaders(params, debug=debug, device=device)

    print('Model device: {}'.format(device))

    pr_loss_fn, yaw_loss_fn, trans_loss_fn, depth_loss_fn = make_losses(params.model_params)

    # Training elements
    if params.weight_decay is None or params.weight_decay == 0:
        if 'ring_sharp' in params.model_params.model:
            optimizer = torch.optim.Adam([{'params':model.parameters()},{'params':trans_cnn.parameters()}], lr=params.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    else:
        if 'ring_sharp' in params.model_params.model:
            optimizer = torch.optim.Adam([{'params':model.parameters()},{'params':trans_cnn.parameters()}], lr=params.lr, weight_decay=params.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    
    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1, eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    radius = [2, 5, 10, 20, 25]

    evaluator_test_set = GLEvaluator(params.dataset_folder, params.dataset, params.test_file, device=device, params=params.model_params, radius=radius, k=20, n_samples=None)

    if do_val:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {e: [] for e in phases}
    stats['eval'] = []
    num_iter = start_iter
    
    for epoch in tqdm.tqdm(range(start_epoch, params.epochs + 1)):
        for phase in phases:
            if 'train' in phase:
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch
            count_batches = 0

            if phase == 'train':
                global_phase = 'global_train'
            elif phase == 'val':
                global_phase = 'global_val'
            f = time.time()
            print(f'------ {phase} ------')
            
            # !!! Below loop will skip some batches in the dataset having larger number of batches
            for batch, positives_mask, negatives_mask, poses, depth_maps in tqdm.tqdm(dataloaders[global_phase]):
                count_batches += 1
                batch_stats = {}
                
                if debug and count_batches > 2:
                    break
                
                # Move everything to the device
                for key in batch.keys():
                    if batch[key] is not None:
                        if key == 'orig_pc':
                            continue
                        batch[key] = batch[key].to(device)

                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()

                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print('WARNING: Skipping batch without positive or negative examples')
                    continue
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y = model(batch)

                    pr_loss = None
                    yaw_loss = None
                    trans_loss = None
                    depth_loss = None
                    pr_stat = {}
                    yaw_stat = {}
                    trans_stat = {}
                    depth_stat = {}

                    if 'ring_sharp' in params.model_params.model:
                        bev = y['bev']
                        spec = y['spec']
                        bev_trans = y['bev_trans']
                        B, C, H, W = bev.shape
                        kl_map = init_kl_map(W, 1)
                        
                        # depth loss
                        if params.model_params.use_depth:
                            depth_preds = y['depth']
                            depth_loss, depth_stat = depth_loss_fn(depth_maps, depth_preds)

                        # # pr loss
                        # pr_loss, pr_stat = pr_loss_fn(spec, positives_mask, negatives_mask)
                        
                        # yaw loss
                        yaw_loss, yaw_stat = yaw_loss_fn(spec, poses, positives_mask, kl_map)
                        
                        # trans loss
                        if bev_trans is not None:
                            trans_loss, trans_stat = trans_loss_fn(bev_trans, poses, positives_mask, kl_map=kl_map, trans_cnn=None)
                        else:
                            trans_loss, trans_stat = trans_loss_fn(bev, poses, positives_mask, kl_map=kl_map, trans_cnn=trans_cnn)
                    else:
                        embedding = y['global']
                        # pr loss
                        pr_loss, pr_stat, _ = pr_loss_fn(embedding, positives_mask, negatives_mask)
                        
                        # yaw loss
                        if 'disco' in params.model_params.model and 'unet_out' in y.keys():
                            unet_out = y['unet_out']
                            B, C, H, W = unet_out.shape
                            kl_map = init_kl_map(W, 1)
                            yaw_loss, yaw_stat = yaw_loss_fn(unet_out, poses, positives_mask, epoch, count_batches, exp_name, kl_map)
                    
                    temp_stats = {**pr_stat, **yaw_stat, **trans_stat, **depth_stat}
                    
                    # total loss
                    total_loss = torch.zeros(1).cuda()
                    if pr_loss is not None:
                        total_loss += pr_loss
                        train_writer.add_scalar('PR_Loss', temp_stats['pr_loss'], num_iter)
                    if yaw_loss is not None:
                        total_loss += yaw_loss
                        train_writer.add_scalar('Yaw_Loss', temp_stats['yaw_loss'], num_iter)
                    if trans_loss is not None:
                        total_loss += trans_loss
                        train_writer.add_scalar('Trans_loss', temp_stats['trans_loss'], num_iter)
                    if depth_loss is not None:
                        total_loss += depth_loss
                        train_writer.add_scalar('Depth_loss', temp_stats['depth_loss'], num_iter)
                    
                    train_writer.add_scalar('Total_loss', total_loss.item(), num_iter)
                    
                    temp_stats = tensors_to_numbers(temp_stats)
                    batch_stats.update(temp_stats)
                    batch_stats['loss'] = total_loss.item()
                    print('batch_stats', batch_stats)

                    num_iter += 1
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                running_stats.append(batch_stats)
                # torch.cuda.empty_cache()  # Prevent excessive GPU memory ddconsumption by SparseTensors
            
            if epoch % params.save_freq == 0 and phase == 'train':
                save_path = os.path.join(weights_path, model_name + '_' + str(epoch) + '.pth')
                if 'ring_sharp' in params.model_params.model:
                    torch.save({'epoch': epoch,
                                'iter': num_iter,
                                'model': model.state_dict(), 
                                'trans_cnn': trans_cnn.state_dict()},
                                save_path)
                else:
                    torch.save({'epoch': epoch,
                                'iter': num_iter,
                                'model': model.state_dict()}, 
                                save_path)          

            # ******* PHASE END *******
            # Compute mean stats for the epoch
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # ******* EPOCH END *******

        if scheduler is not None:
            scheduler.step()

        train_writer.add_scalar('Train_loss', stats['train'][-1]['loss'], epoch)
        print('Train loss: ', stats['train'][-1]['loss'])
        if do_val:
            train_writer.add_scalar('Val_loss', stats['val'][-1]['loss'], epoch)
            print('Val loss: ', stats['val'][-1]['loss'])

        metrics = {'train': {}, 'val': {}, 'test': {}}
        metrics['train']['loss'] = stats['train'][-1]['loss']
        if do_val:
            metrics['val']['loss'] = stats['val'][-1]['loss']

        if epoch % params.eval_freq == 0:
            print('------ eval ------')
            if 'ring_sharp' in params.model_params.model:
                global_metrics = evaluator_test_set.evaluate(model, trans_cnn, exp_name)
            else:
                global_metrics = evaluator_test_set.evaluate(model, exp_name)
            evaluator_test_set.print_results(global_metrics)
            evaluator_test_set.export_eval_stats(f'{weights_path}/eval_results_{epoch}epoch.txt', global_metrics)
            # Write to tensorboard
            for key in global_metrics.keys():
                val = global_metrics[key]
                if isinstance(val, dict):
                    for k in val.keys():
                        if isinstance(val[k], (list, np.ndarray)):
                            train_writer.add_scalar(f'Test_{key}_{k}', val[k][0], epoch)
                elif isinstance(val, (list, np.ndarray)):
                    train_writer.add_scalar(f'Test_{key}', val[0], epoch)
                elif isinstance(val, (float, int)):
                    train_writer.add_scalar(f'Test_{key}', val, epoch)
                else:
                    continue
        
        if params.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' not in epoch_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['global_train'].batch_sampler.expand_batch()

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    print('')

    # Save final model weights
    final_model_path = os.path.join(weights_path, model_name + '_final.pth')
    if 'ring_sharp' in params.model_params.model:
        torch.save({'epoch':epoch,
                    'iter':num_iter,
                    'model':model.state_dict(), 
                    'trans_cnn':trans_cnn.state_dict()},
                    final_model_path)
    else:
        torch.save({'epoch':epoch,
                    'iter':num_iter,
                    'model':model.state_dict()}, 
                    final_model_path)
    
    # Evaluate the final model using all samples
    if 'ring_sharp' in params.model_params.model:
        global_metrics = evaluator_test_set.evaluate(model, trans_cnn, exp_name)
    else:
        global_metrics = evaluator_test_set.evaluate(model, exp_name)    
    evaluator_test_set.print_results(global_metrics)
    evaluator_test_set.export_eval_stats(f'{weights_path}/final_eval_results.txt', global_metrics)
