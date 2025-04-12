# Zhejiang University

import os
import configparser
import time
import numpy as np
import torch
from glnet.datasets.quantization import PolarQuantizer, CartesianQuantizer, PassThroughQuantizer, EgoNNPolarQuantizer

class ModelParams:
    def __init__(self, model_params_path, dataset_type, dataset_folder):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.dataset_folder = os.path.expanduser(dataset_folder)
        self.dataset_type = dataset_type
        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.use_rgb = params.getboolean('use_rgb', False)
        self.use_mask = params.getboolean('use_mask', False)
        self.use_depth = params.getboolean('use_depth', False)
        self.use_pretrained_model = params.getboolean('use_pretrained_model', False)

        self.use_xyz = params.getboolean('use_xyz', False)
        self.use_bev = params.getboolean('use_bev', False)
        self.use_submap = params.getboolean('use_submap', False)
        self.use_panorama = params.getboolean('use_panorama', False)
        self.use_range_image = params.getboolean('use_range_image', False)
        self.xyz_aug = params.getboolean('xyz_aug', False)
        self.deform_attn = params.getboolean('deform_attn', False)
        self.normalize = params.getboolean('normalize', True)
        self.confidence = params.getboolean('confidence', False)

        self.aggregation = params.get('aggregation','gem').lower()
        self.lidar_fix_num = params.getint('lidar_fix_num', 20000)
        self.theta = params.getint('theta', 120)
        self.radius = params.getint('radius', 40)
        self.Z = params.getint('Z', 120)      # Size of BEV, Z is front, Y is down, X is right
        self.Y = params.getint('Y', 20)
        self.X = params.getint('X', 120) 
        self.H = params.getint('Height', 240)
        self.W = params.getint('Width', 360)

        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor
        self.feature_dim = params.getint('feature_dim', 256)      # Size of the final descriptor
        self.voxel_num_points = params.getint('voxel_num_points', 55)

        self.loss = params.get('loss', 'BatchHardTripletMarginLoss')

        if 'cam_id' in params:
            self.cam_id = [int(e) for e in params['cam_id'].split(',')]
        else: 
            self.cam_id = None

        if 'scene_centroid' in params:
            self.scene_centroid_py = [float(e) for e in params['scene_centroid'].split(',')]
            self.scene_centroid = torch.from_numpy(np.array(self.scene_centroid_py).reshape([1, 3])).float()
        else: 
            self.scene_centroid = None
            
        if 'xbounds' in params and 'ybounds' in params and 'zbounds' in params:
            self.xbounds = [float(e) for e in params['xbounds'].split(',')]
            self.ybounds = [float(e) for e in params['ybounds'].split(',')]
            self.zbounds = [float(e) for e in params['zbounds'].split(',')]
            XMIN = self.xbounds[0]
            XMAX = self.xbounds[1]
            YMIN = self.ybounds[0]
            YMAX = self.ybounds[1]        
            ZMIN = self.zbounds[0]
            ZMAX = self.zbounds[1]
            self.bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
            print('bounds: ', self.bounds)
        else:
            self.xbounds = self.ybounds = self.zbounds = None
            self.bounds = None

        if 'image_meta_path' in params:
            self.image_meta_path = params.get('image_meta_path', os.path.expanduser('~/Data/NCLT/image_meta.pkl'))
        else:
            self.image_meta_path = None

        if 'coordinates' in params:
            self.coordinates = params.get('coordinates', 'polar')
            assert self.coordinates in ['polar', 'cartesian', 'passthrough', 'None'], f'Unsupported coordinates: {self.coordinates}'
        else:
            self.coordinates = 'None'

        if 'polar' in self.coordinates and 'egonn' in self.model:
            # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
            self.quantization_step = [float(e) for e in params['quantization_step'].split(',')]
            assert len(self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
            self.quantizer = EgoNNPolarQuantizer(quant_step=self.quantization_step)
        elif 'polar' in self.coordinates:
            # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
            assert self.bounds is not None, 'param: xbounds, ybounds, zbounds should be set'
            self.quantization_step = [XMAX/self.radius, (YMAX-YMIN)/self.Y, 360./self.theta]
            assert len(self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
            self.quantizer = PolarQuantizer(self.image_meta_path, quant_step=self.quantization_step)
            self.xbounds[0] = 0.
            self.zbounds[1] = 360.
            self.zbounds[0] = 0.
        elif 'cartesian' in self.coordinates:
            # Single quantization step for cartesian coordinates
            self.quantization_step = [(XMAX-XMIN)/self.X, (ZMAX-ZMIN)/self.Z, (YMAX-YMIN)/self.Y]
            self.quantizer = CartesianQuantizer(self.image_meta_path, quant_step=self.quantization_step)
        elif 'passthrough' in self.coordinates:
            self.quantizer = PassThroughQuantizer()
        elif 'None' in self.coordinates:
            self.quantization_step = None
            self.quantizer = None
        else:
            raise NotImplementedError(f'Unsupported coordinates: {self.coordinates}')

        if 'image_encoder' in params:
            self.image_encoder = params.get('image_encoder', 'res18')
        else:
            self.image_encoder = None

        if 'bev_encoder' in params:
            self.bev_encoder = params.get('bev_encoder', 'res18')
        else:
            self.bev_encoder = None
        
        if 'point_encoder' in params:
            self.point_encoder = params.get('point_encoder', 'voxel')
        else:
            self.point_encoder = None

        print(f'---------------- Using {self.loss} Loss Function ----------------')        
        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)    # Margin used in loss function
        elif 'Overlap' in self.loss:
            self.use_overlap = True
    
    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e is not None:
                print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime('%Y%m%d_%H%M')


class TrainingParams:
    """
    Parameters for model training
    """
    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset = params.get('dataset', 'nclt').lower()
        assert self.dataset in ['nclt', 'oxford'], '{} is not supported'.format(self.dataset)
        self.dataset_folder = os.path.expanduser(params.get('dataset_folder'))

        # Maximum random rotation and translation applied when generating pairs for local descriptor
        self.rot_max = params.getfloat('rot_max', np.pi)
        self.trans_max = params.getfloat('rot_max', 5.)

        params = config['TRAIN']

        self.save_freq = params.getint('save_freq', 1)          # Model saving frequency (in epochs)
        self.eval_freq = params.getint('eval_freq', 2)          # Model evaluation frequency (in epochs)

        self.num_workers = params.getint('num_workers', 4)
        self.batch_size = params.getint('batch_size', 64)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate >= 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = params.getfloat('lr', 1e-3)

        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.use_overlap = False
            
        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.test_file = params.get('test_file', None)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path, self.dataset, self.dataset_folder)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))
        
        self.model_params.print()
        print('')

