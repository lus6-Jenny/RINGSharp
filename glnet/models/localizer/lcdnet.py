from __future__ import print_function
import os
import torch
import torch.nn as nn
import numpy as np

from pcdet.config import cfg_from_yaml_file
from pcdet.config import cfg as pvrcnn_cfg
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.processor.data_processor import DataProcessor

from glnet.utils.params import ModelParams
from glnet.models.backbone3D.PVRCNN import PVRCNN
from glnet.models.backbone3D.NetVlad import NetVLADLoupe
from glnet.models.backbone3D.models_3d import NetVladCustom
import yaml

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
grandparent_directory = os.path.dirname(parent_directory)

wandb_config_filepath = os.path.join(grandparent_directory, 'config/wandb_config.yaml')
pv_rcnn_filepath = os.path.join(grandparent_directory, 'models/backbone3D/pv_rcnn.yaml')


class LCDNet(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(LCDNet, self).__init__()
        with open(wandb_config_filepath, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            
        self.exp_cfg = cfg['experiment']
        rotation_parameters = 1
        self.exp_cfg['use_svd'] = True

        if self.exp_cfg['3D_net'] == 'PVRCNN':
            cfg_from_yaml_file(pv_rcnn_filepath, pvrcnn_cfg)
            pvrcnn_cfg.MODEL.PFE.NUM_KEYPOINTS = self.exp_cfg['num_points']
            if 'PC_RANGE' in self.exp_cfg:
                pvrcnn_cfg.DATA_CONFIG.POINT_CLOUD_RANGE = self.exp_cfg['PC_RANGE']
            pvrcnn = PVRCNN(pvrcnn_cfg, True, self.exp_cfg['model_norm'], self.exp_cfg['shared_embeddings'])
            net_vlad = NetVLADLoupe(feature_size=pvrcnn_cfg.MODEL.PFE.NUM_OUTPUT_FEATURES,
                                    cluster_size=self.exp_cfg['cluster_size'],
                                    output_dim=self.exp_cfg['feature_output_dim_3D'],
                                    gating=True, add_norm=True, is_training=True)
            model = NetVladCustom(pvrcnn, net_vlad, feature_norm=False, fc_input_dim=640,
                                points_num=self.exp_cfg['num_points'], head=self.exp_cfg['head'],
                                rotation_parameters=rotation_parameters, sinkhorn_iter=self.exp_cfg['sinkhorn_iter'],
                                use_svd=self.exp_cfg['use_svd'], sinkhorn_type=self.exp_cfg['sinkhorn_type'])
        else:
            raise TypeError("Unknown 3D network")
        self.model = model
        point_cloud_range = np.array(pvrcnn_cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        self.data_processor = DataProcessor(pvrcnn_cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range, True, 4)

    def prepare_input(self, point_cloud):
        batch_dict = {'points': point_cloud.cpu().numpy(), 'use_lead_xyz': True}
        batch_dict = self.data_processor.forward(batch_dict)
        return batch_dict

    def forward(self, batch_dict, metric_head):
        anchor_list = []
        positive_list = []

        if metric_head == False:
            anchor_i_list = batch_dict['coords']
            
            for i in range(len(anchor_i_list)):
                anchor_i = anchor_i_list[i].cuda()
                anchor_i_reflectance = torch.ones(anchor_i.shape[0],1).float().to(anchor_i.device)
                anchor_i = torch.cat((anchor_i, anchor_i_reflectance), 1)
                anchor_i[:, 3] = 1.
                anchor_list.append(self.prepare_input(anchor_i))
                del anchor_i
            model_in = KittiDataset.collate_batch(anchor_list + positive_list)

            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().cuda()

            compute_embeddings = True
            compute_transl = False
            compute_rotation = False
            # print("model_in", model_in)
            batch_dict = self.model(model_in, metric_head, compute_embeddings,
                            compute_transl, compute_rotation, mode='pairs')

            batch_dict["global"] = batch_dict.pop("out_embedding")
            batch_dict["global"] = batch_dict["global"] / batch_dict["global"].norm(dim=1, keepdim=True)

            return batch_dict

        else:
            anchor_i_list = batch_dict['anc_pcd']
            positive_i_list = batch_dict['pos_pcd']

            for i in range(len(anchor_i_list)):
                anchor_i = anchor_i_list[i].cuda()
                anchor_i_reflectance = torch.ones(anchor_i.shape[0],1).float().to(anchor_i.device)
                anchor_i = torch.cat((anchor_i, anchor_i_reflectance), 1)
                anchor_i[:, 3] = 1.
                anchor_list.append(self.prepare_input(anchor_i))
                del anchor_i
            for i in range(len(positive_i_list)):
                positive_i = positive_i_list[i].cuda()
                positive_i_reflectance = torch.ones(positive_i.shape[0],1).float().to(positive_i.device)
                positive_i = torch.cat((positive_i, positive_i_reflectance), 1)
                positive_i[:, 3] = 1.
                anchor_list.append(self.prepare_input(positive_i))
                del positive_i
            # end = time.time()
            # print(end-start)
            model_in = KittiDataset.collate_batch(anchor_list + positive_list)

            for key, val in model_in.items():
                if not isinstance(val, np.ndarray):
                    continue
                model_in[key] = torch.from_numpy(val).float().cuda()

            compute_embeddings = True
            compute_transl = True
            compute_rotation = True
            # print("model_in", model_in)
            batch_dict = self.model(model_in, metric_head, compute_embeddings,
                            compute_transl, compute_rotation, mode='pairs')

            batch_dict["global"] = batch_dict.pop("out_embedding")
            batch_dict["global"] = batch_dict["global"] / batch_dict["global"].norm(dim=1, keepdim=True)

            return batch_dict

    def print_info(self):
        print('Model class: LCDNet')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))