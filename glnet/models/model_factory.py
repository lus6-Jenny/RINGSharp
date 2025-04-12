
# Visual Localization Network
from glnet.models.localizer.mixvpr import MixVPRModel
from glnet.models.localizer.anyloc import AnyLocWrapper
from glnet.models.localizer.sfrs import SFRSModel
from RINGSharp.glnet.models.localizer.ring_sharp_v import RINGSharpV
from glnet.models.localizer.vdisco import vDiSCO
from glnet.models.localizer.netvlad import NetVLAD, NetVLAD_Pretrain
from glnet.models.localizer.patch_netvlad import PatchNetVLAD_Pretrain

# LiDAR Localization Network
from glnet.models.localizer.ring import RING
from glnet.models.localizer.ring_plus_plus import RINGPlusPlus
from glnet.models.localizer.ring_sharp_l import RINGSharpL
from glnet.models.localizer.disco import DiSCO
from glnet.models.localizer.lcdnet import LCDNet
from glnet.models.localizer.overlap_transformer import OverlapTransformer
from glnet.models.backbones_2d.eca_block import ECABasicBlock
from glnet.models.localizer.minkgl import MinkHead, MinkTrunk, MinkGL

from glnet.utils.params import ModelParams
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def model_factory(model_params: ModelParams):
    if 'mixvpr' in model_params.model:
        model = MixVPRModel(model_params)
    elif 'anyloc' in model_params.model:
        model = AnyLocWrapper()
    elif 'sfrs' in model_params.model:
        model = SFRSModel()
    elif 'ring_plus_plus' in model_params.model:
        model = RINGPlusPlus(model_params)
    elif 'ring' in model_params.model and 'ring_sharp' not in model_params.model:
        model = RING(model_params)
    elif 'ring_sharp_v' in model_params.model:
        model = RINGSharpV(model_params)
    elif 'ring_sharp_l' in model_params.model:
        model = RINGSharpL(model_params)
    elif 'vdisco' in model_params.model:
        model = vDiSCO(model_params)
    elif 'netvlad' in model_params.model and 'pretrain' not in model_params.model:
        model = NetVLAD(model_params)
    elif 'netvlad_pretrain' in model_params.model and 'patch' not in model_params.model:
        model = build_netvlad_pretrain(model_params)
    elif 'patch_netvlad_pretrain' in model_params.model:
        model = build_patch_netvlad_pretrain(model_params)
    elif 'disco' in model_params.model:
        model = DiSCO(model_params)
    elif 'lcdnet' in model_params.model:
        model = LCDNet(model_params)
    elif 'overlaptransformer' in model_params.model:
        model = OverlapTransformer(model_params)
    elif 'egonn' in model_params.model:
        model = create_egonn_model(model_params)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(model_params.model))
    
    return model


def create_egonn_model(model_params: ModelParams):
    global_normalize = False
    local_normalize = True

    # This is the best model of EgoNN
    block = ECABasicBlock
    planes = [32, 64, 64, 128, 128, 128, 128]
    layers = [1, 1, 1, 1, 1, 1, 1]

    global_in_levels = [5, 6, 7]
    global_map_channels = 128
    global_descriptor_size = 256

    local_in_levels = [3, 4]
    local_map_channels = 64
    local_descriptor_size = 128
    
    # Planes list number of channels for level 1 and above
    global_in_channels = [planes[i-1] for i in global_in_levels]
    head_global = MinkHead(global_in_levels, global_in_channels, global_map_channels)

    if len(local_in_levels) > 0:
        local_in_channels = [planes[i-1] for i in local_in_levels]
        head_local = MinkHead(local_in_levels, local_in_channels, local_map_channels)
    else:
        head_local = None

    min_out_level = len(planes)
    if len(global_in_levels) > 0:
        min_out_level = min(min_out_level, min(global_in_levels))
    if len(local_in_levels) > 0:
        min_out_level = min(min_out_level, min(local_in_levels))

    trunk = MinkTrunk(in_channels=1, planes=planes, layers=layers, conv0_kernel_size=5, block=block,
                      min_out_level=min_out_level)

    net = MinkGL(trunk, local_head=head_local, local_descriptor_size=local_descriptor_size,
                 local_normalize=local_normalize, global_head=head_global,
                 global_descriptor_size=global_descriptor_size, global_pool_method='GeM',
                 global_normalize=global_normalize, quantizer=model_params.quantizer)

    return net


class NetVLAD_finetune(nn.Module):
    def __init__(self, encoder, pool, model_params):
        super(NetVLAD_finetune, self).__init__()
        self.encoder = encoder
        self.pool = pool
        self.model_params = model_params

    def forward(self, batch):
        x = batch['img']
        if not self.model_params.use_panorama:
            x = x[:,0,...] # use front-view camera image as input
                
        x = self.encoder(x)
        x = self.pool(x)
        
        return {'global': x}
    
    def print_info(self):   
        print('Model class: NetVLAD')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.encoder.parameters()])
        print('Image Backbone parameters: {}'.format(n_params))


def build_netvlad_pretrain(model_params):

    print('===> Building NetVLAD model')
    encoder_dim = 512
    num_clusters = 64
    encoder = models.vgg16(pretrained=True)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    for l in layers[:-1]:
        for p in l.parameters():
            p.requires_grad = False
    layers.append(L2Norm())
    encoder = nn.Sequential(*layers)
    net_vlad = NetVLAD_Pretrain(num_clusters=64, dim=encoder_dim, vladv2=False)

    model = NetVLAD_finetune(encoder, net_vlad, model_params)

    return model
        

class PatchNetVLAD_finetune(nn.Module):
    def __init__(self, encoder, pool, model_params):
        super(PatchNetVLAD_finetune, self).__init__()
        self.encoder = encoder
        self.pool = pool
        self.model_params = model_params
    
    def forward(self, batch):
        x = batch['img']
        if not self.model_params.use_panorama:
            x = x[:,0,...] # use front-view camera image as input

        x = self.encoder(x)
        x = self.pool(x)
        return {'global': x}
    
    def print_info(self):   
        print('Model class: Patch-NetVLAD')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.encoder.parameters()])
        print('Image Backbone parameters: {}'.format(n_params))


def build_patch_netvlad_pretrain(model_params):

    print('===> Building Patch NetVLAD model')
    encoder_dim = 512
    num_clusters = 64
    encoder = models.vgg16(pretrained=True)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    for l in layers[:-1]:
        for p in l.parameters():
            p.requires_grad = False
    layers.append(L2Norm())
    encoder = nn.Sequential(*layers)
    patch_netvlad = PatchNetVLAD_Pretrain(num_clusters=64, dim=encoder_dim, vladv2=False)

    model = PatchNetVLAD_finetune(encoder, patch_netvlad, model_params)

    return model