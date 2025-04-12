# Code from AnyLoc https://arxiv.org/abs/2308.00688

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Union, List, Tuple, Literal
from glnet.models.aggregation.GeM import GeM

device = "cuda" if torch.cuda.is_available() else "cpu"

# # %% -------------------- Dino-v2 utilities --------------------
# # Extract features from a Dino-v2 model
# _DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
#                         "dinov2_vitl14", "dinov2_vitg14"]
# _DINO_FACETS = Literal["query", "key", "value", "token"]
# class DinoV2ExtractFeatures:
#     """
#         Extract features from an intermediate layer in Dino-v2
#     """
#     def __init__(self, dino_model: _DINO_V2_MODELS, layer: int, 
#                 facet: _DINO_FACETS="token", use_cls=False, 
#                 norm_descs=True, device: str = "cpu") -> None:
#         """
#             Parameters:
#             - dino_model:   The DINO-v2 model to use
#             - layer:        The layer to extract features from
#             - facet:    "query", "key", or "value" for the attention
#                         facets. "token" for the output of the layer.
#             - use_cls:  If True, the CLS token (first item) is also
#                         included in the returned list of descriptors.
#                         Otherwise, only patch descriptors are used.
#             - norm_descs:   If True, the descriptors are normalized
#             - device:   PyTorch device to use
#         """
#         self.vit_type: str = dino_model
#         self.dino_model: nn.Module = torch.hub.load(
#                 'facebookresearch/dinov2', dino_model)
#         self.device = torch.device(device)
#         self.dino_model = self.dino_model.eval().to(self.device)
#         self.layer: int = layer
#         self.facet = facet
#         if self.facet == "token":
#             self.fh_handle = self.dino_model.blocks[self.layer].\
#                     register_forward_hook(
#                             self._generate_forward_hook())
#         else:
#             self.fh_handle = self.dino_model.blocks[self.layer].\
#                     attn.qkv.register_forward_hook(
#                             self._generate_forward_hook())
#         self.use_cls = use_cls
#         self.norm_descs = norm_descs
#         # Hook data
#         self._hook_out = None
    
#     def _generate_forward_hook(self):
#         def _forward_hook(module, inputs, output):
#             self._hook_out = output
#         return _forward_hook
    
#     def __call__(self, img: torch.Tensor) -> torch.Tensor:
#         """
#             Parameters:
#             - img:   The input image
#         """
#         with torch.no_grad():
#             res = self.dino_model(img)
#             if self.use_cls:
#                 res = self._hook_out
#             else:
#                 res = self._hook_out[:, 1:, ...]
#             if self.facet in ["query", "key", "value"]:
#                 d_len = res.shape[2] // 3
#                 if self.facet == "query":
#                     res = res[:, :, :d_len]
#                 elif self.facet == "key":
#                     res = res[:, :, d_len:2*d_len]
#                 else:
#                     res = res[:, :, 2*d_len:]
#         if self.norm_descs:
#             res = F.normalize(res, dim=-1)
#         self._hook_out = None   # Reset the hook
#         return res
    
#     def __del__(self):
#         self.fh_handle.remove()
        

# class AnyLocWrapper(torch.nn.Module):
#     default_config = {
#         "dino_model": "dinov2_vitg14", # Model type
#         "layer": 31, # Layer for extracting Dino feature (descriptors)
#         "facet": "token", # Facet for extracting descriptors
#         # "aggregation": "VLAD",
#         "gem_use_abs": False,
#         "gem_elem_by_elem": False, # Do GeM element-by-element (only if gem_use_abs = False)
#         "gem_p": 3, # GeM Pooling Parameter
#     }    
#     def __init__(self, config=default_config):
#         super().__init__()
#         self.config = config
#         self.dinov2 =  DinoV2ExtractFeatures(dino_model=config["dino_model"], layer=config["layer"], facet=config["facet"], device=device)
        
#     def forward(self, batch):
#         images = batch['img']
#         if images.dim() == 5:
#             B, S, C, H, W = images.shape
#             images = images.view(B*S, C, H, W)
#         else:
#             B, C, H, W = images.shape
#         # DINO wants height and width as multiple of 14, therefore resize them
#         # to the nearest multiple of 14
#         H = round(H / 14) * 14
#         W = round(W / 14) * 14
#         images = transforms.functional.resize(images, [H, W], antialias=True)
#         patch_descs = self.dinov2(images)
#         print('DINOv2 feats shape', patch_descs.shape)
#         x = self.get_gem_descriptors(patch_descs)
#         print('Global descriptors shape after GeM', x.shape)
        
#         return {'global': x}
    
#     def get_gem_descriptors(self, patch_descs: torch.Tensor):
#         assert len(patch_descs.shape) == len(("N", "n_p", "d_dim"))
#         g_res = None
#         if self.config['gem_use_abs']:
#             g_res = torch.mean(torch.abs(patch_descs)**self.config['gem_p'], 
#                     dim=-2) ** (1/self.config['gem_p'])
#         else:
#             if self.config['gem_elem_by_elem']:
#                 g_res_all = []
#                 for patch_desc in patch_descs:
#                     x = torch.mean(patch_desc**self.config['gem_p'], dim=-2)
#                     g_res = x.to(torch.complex64) ** (1/self.config['gem_p'])
#                     g_res = torch.abs(g_res) * torch.sign(x)
#                     g_res_all.append(g_res)
#                 g_res = torch.stack(g_res_all)
#             else:
#                 x = torch.mean(patch_descs**self.config['gem_p'], dim=-2)
#                 g_res = x.to(torch.complex64) ** (1/self.config['gem_p'])
#                 g_res = torch.abs(g_res) * torch.sign(x)
#         return g_res    # [N, d_dim]  


class AnyLocWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", device="cuda")
        
    def forward(self, batch):
        images = batch['img']
        if images.dim() == 5:
            B, S, C, H, W = images.shape
            images = images.view(B*S, C, H, W)
        else:
            B, C, H, W = images.shape
        # DINO wants height and width as multiple of 14, therefore resize them
        # to the nearest multiple of 14
        H = round(H / 14) * 14
        W = round(W / 14) * 14
        images = transforms.functional.resize(images, [H, W], antialias=True)
        x = self.model(images)
        return {'global': x}

    def print_info(self):
        print('debug', self.model)
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)        
            param.requires_grad = True        
        print('Model class: AnyLoc')
        n_params = sum([param.nelement() for param in self.model.parameters()])
        print('Total parameters: {}'.format(n_params))
        