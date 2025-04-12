# Unified model to generate a global and local descriptors
# Warsaw University of Technology

from typing import Dict, List
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock

import glnet.models.aggregation.pooling as pooling
from glnet.datasets.quantization import Quantizer


class MinkHead(nn.Module):
    def __init__(self, in_levels: List[int], in_channels: List[int], out_channels: int):
        # in_levels: list of levels from the bottom_up trunk which output is taken as an input (starting from 0)
        # in_channels: number of channels in each input_level
        # out_channels: number of output channels
        assert len(in_levels) > 0
        assert len(in_levels) == len(in_channels)

        super().__init__()
        self.in_levels = in_levels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.min_level = min(in_levels)
        self.max_level = max(in_levels)
        assert self.min_level > 0

        # Dictionary in_d[input_level] = number of channels
        self.in_d = {in_level: in_channel for in_level, in_channel in zip(self.in_levels, self.in_channels)}

        self.conv1x1 = nn.ModuleDict()   # 1x1 convolutions in lateral connections
        self.tconv = nn.ModuleDict()    # Top-down transposed convolutions

        for in_level in range(self.min_level+1, self.max_level+1):
            # There's one transposed convolution for each level, except for min_level
            self.tconv[str(in_level)] = ME.MinkowskiConvolutionTranspose(self.out_channels, self.out_channels,
                                                                         kernel_size=2, stride=2, dimension=3)
        for in_level in self.in_d:
            # Create lateral 1x1x convolution for each input level
            self.conv1x1[str(in_level)] = ME.MinkowskiConvolution(self.in_d[in_level], self.out_channels, kernel_size=1,
                                                                  stride=1, dimension=3)

    def forward(self, x: Dict[int, ME.SparseTensor]) -> ME.SparseTensor:
        # x: dictionary of tensors from different trunk levels, indexed by the trunk level

        # TOP-DOWN PASS
        y = self.conv1x1[str(self.max_level)](x[self.max_level])
        #y = x[self.max_level]
        for level in range(self.max_level-1, self.min_level-1, -1):
            y = self.tconv[str(level+1)](y)
            if level in self.in_d:
                # Add the feature map from the corresponding level at bottom-up pass using a skip connection
                y = y + self.conv1x1[str(level)](x[level])

        assert y.shape[1] == self.out_channels

        return y

    def __str__(self):
        n_params = sum([param.nelement() for param in self.parameters()])
        str = f'Input levels: {self.in_levels}   Input channels: {self.in_channels}   #parameters: {n_params}'
        return str


class MinkTrunk(nn.Module):
    # Bottom-up trunk

    def __init__(self, in_channels: int, planes: List[int], layers: List[int] = None,
                 conv0_kernel_size: int = 5, block=BasicBlock, min_out_level: int = 1):
        # min_out_level: minimum level of bottom-up pass to return the feature map. If None, all feature maps from
        #                the level 1 to the top are returned.

        super().__init__()
        self.in_channels = in_channels
        self.planes = planes
        if layers is None:
            self.layers = [1] * len(self.planes)
        else:
            assert len(layers) == len(planes)
            self.layers = layers

        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        assert min_out_level <= len(planes)
        self.min_out_level = min_out_level

        self.num_bottom_up = len(self.planes)
        self.init_dim = planes[0]
        assert len(self.layers) == len(self.planes)

        self.convs = nn.ModuleDict()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleDict()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleDict()  # Bottom-up blocks

        # The first convolution is special case, with larger kernel
        self.inplanes = self.planes[0]
        self.convs[str(0)] = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                                     dimension=3)
        self.bn[str(0)] = ME.MinkowskiBatchNorm(self.inplanes)

        for ndx, (plane, layer) in enumerate(zip(self.planes, self.layers)):
            self.convs[str(ndx + 1)] = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2,
                                                               dimension=3)
            self.bn[str(ndx + 1)] = ME.MinkowskiBatchNorm(self.inplanes)
            self.blocks[str(ndx + 1)] = self._make_layer(self.block, plane, layer)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.weight_initialization()

    def weight_initialization(self):
            for m in self.modules():
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1 ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(ME.MinkowskiConvolution(self.inplanes, planes * block.expansion, kernel_size=1,
                                                               stride=stride, dimension=3),
                                       ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample,
                            dimension=3))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=3))

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor) -> Dict[int, ME.SparseTensor]:
        # *** BOTTOM-UP PASS ***
        y = {}      # Output feature maps

        x = self.convs[str(0)](x)
        x = self.bn[str(0)](x)
        x = self.relu(x)

        # BOTTOM-UP PASS
        for i in range(1, len(self.layers)+1):
            x = self.convs[str(i)](x)           # Decreases spatial resolution (conv stride=2)
            x = self.bn[str(i)](x)
            x = self.relu(x)
            x = self.blocks[str(i)](x)
            if i >= self.min_out_level:
                y[i] = x

        return y

    def __str__(self):
        n_params = sum([param.nelement() for param in self.parameters()])
        str = f'Planes: {self.planes}   Min out Level: {self.min_out_level}   #parameters: {n_params}'
        return str


class SaliencyRegressor(nn.Module):
    # Regress 1-channel saliency map with values in <0,1> range from the input in_channels feature map
    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.net = nn.Sequential(ME.MinkowskiLinear(in_channels, in_channels // reduction),
                                 ME.MinkowskiReLU(inplace=True), ME.MinkowskiLinear(in_channels // reduction, 1),
                                 ME.MinkowskiSigmoid())

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.net(x)


class KeypointRegressor(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.net = nn.Sequential(ME.MinkowskiLinear(in_channels, in_channels // reduction),
                                 ME.MinkowskiReLU(inplace=True), ME.MinkowskiLinear(in_channels // reduction, 3),
                                 ME.MinkowskiTanh())

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.net(x)


class SigmaRegressor(nn.Module):
    # Keypoint uncertainty regressor
    # Returns saliency map with values in 0..infinity range
    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.sigma_lower_bound = 1e-6
        self.net = nn.Sequential(ME.MinkowskiLinear(in_channels, in_channels // reduction),
                                 ME.MinkowskiReLU(inplace=True), ME.MinkowskiLinear(in_channels // reduction, 1),
                                 ME.MinkowskiSoftplus())

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        x = self.net(x)
        # TODO: Below line is not working with sparse tensors. Fix it.
        #return x + self.sigma_lower_bound
        return x


class DescriptorDecoder(nn.Module):
    # Keypoint descriptor decoder
    def __init__(self, in_channels: int, out_channels: int, normalize=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        middle_channels = out_channels + (in_channels - out_channels) // 2
        self.normalize = normalize
        self.net = nn.Sequential(ME.MinkowskiLinear(in_channels, middle_channels),
                                 ME.MinkowskiReLU(inplace=True),
                                 ME.MinkowskiLinear(middle_channels, out_channels))

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        x = self.net(x)

        if self.normalize:
            x = ME.MinkowskiFunctional.normalize(x)

        return x


class MinkGL(nn.Module):
    def __init__(self, trunk: MinkTrunk,
                 local_head: MinkHead = None, local_descriptor_size: int = None, local_normalize: bool = True,
                 global_head: MinkHead = None, global_descriptor_size: int = None, global_pool_method: str ='GeM',
                 global_normalize: bool = False, quantizer: Quantizer = None):

        # quantizer: needed for local descriptor to convert keypoint coordinates into real-value cartesian coords
        assert quantizer is not None

        super().__init__()

        self.trunk = trunk
        self.global_head = global_head
        self.global_pool_method = global_pool_method
        self.global_channels = self.global_head.out_channels
        self.global_pooling = pooling.PoolingWrapper(pool_method=self.global_pool_method,
                                                     in_dim=self.global_channels, output_dim=self.global_channels)
        self.global_normalize = global_normalize
        self.global_descriptor_size = global_descriptor_size
        n_channels = self.global_head.out_channels
        self.global_descriptor_decoder = DescriptorDecoder(n_channels, self.global_descriptor_size, normalize=False)

        if local_head is None:
            self.local_head = None
        else:
            self.local_head = local_head
            self.local_descriptor_size = local_descriptor_size
            self.local_normalize = local_normalize
            n_channels = self.local_head.out_channels

            self.local_keypoint_regressor = KeypointRegressor(n_channels, reduction=2)
            self.local_sigma_regressor = SigmaRegressor(n_channels, reduction=2)
            self.local_descriptor_decoder = DescriptorDecoder(n_channels, self.local_descriptor_size,
                                                              normalize=local_normalize)
        self.quantizer = quantizer
        # Disables keypoint position regresor and assumes keypoint coordinates at the supervoxel centres
        # used for ablation study
        self.ignore_keypoint_regressor = False

    def forward(self, batch: Dict[str, torch.tensor], disable_global_head: bool = False,
                disable_local_head: bool = False):
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        x = self.trunk(x)
        y = {}

        if not disable_global_head:
            x_global = self.global_head(x)
            x_global = self.global_descriptor_decoder(x_global)
            if self.global_normalize:
                x_global = ME.MinkowskiFunctional.normalize(x_global)

            x_global = self.global_pooling(x_global)        # (batch_size, features) tensor

            assert x_global.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got ' \
                                        f'{x_global.dim()} dimensions.'
            assert x_global.shape[1] == self.global_descriptor_size, f"Descriptor has {x_global.shape[1]}. Expected " \
                                                                     f"{self.global_descriptor_size} channels."
            y['global'] = x_global

        if self.local_head is not None and not disable_local_head:
            x_local = self.local_head(x)
            x_descriptors = self.local_descriptor_decoder(x_local)
            x_descriptors = x_descriptors.decomposed_features
            y['descriptors'] = x_descriptors

            kp_offset = self.local_keypoint_regressor(x_local)
            assert kp_offset.shape[1] == 3

            if self.ignore_keypoint_regressor:
                # Assume keypoint position at supervoxel centres
                kp_pos = self.quantizer.keypoint_position(kp_offset.C[:, 1:], kp_offset.tensor_stride,
                                                          torch.zeros_like(kp_offset.F))
            else:
                # Compute cartesian coordinates based on supervoxel center, supervoxel stride and keypoint offset
                kp_pos = self.quantizer.keypoint_position(kp_offset.C[:, 1:], kp_offset.tensor_stride, kp_offset.F)

            y['keypoints'] = [kp_pos[ndx] for ndx in kp_offset._batchwise_row_indices]
            x_sigma = self.local_sigma_regressor(x_local)
            assert x_sigma.shape[1] == 1
            x_sigma = x_sigma.decomposed_features
            y['sigma'] = x_sigma

        # kp_pos is a list of (n_keypoints, 3) tensors with keypoint coordinates in a world coordinate frame
        # sigma is a list of (n_keypoints, 1) tensors with keypoint uncertainties
        # kp_descriptor is a list of (n_keypoints, 256) tensors with keypoint descriptors
        # Each batch element is a separate item in these lists

        return y

    def print_info(self):
        print(f'Model class: {type(self).__name__}')
        n_params = sum([param.nelement() for param in self.parameters()])
        n_trunk_params = sum([param.nelement() for param in self.trunk.parameters()])
        print(f"# parameters - total: {n_params/1000:.1f}   trunk: {n_trunk_params/1000:.1f} [k]")
        if self.local_head is not None:
            n_local_head = sum([param.nelement() for param in self.local_head.parameters()])
            n_keypoint_regressor = sum([param.nelement() for param in self.local_keypoint_regressor.parameters()])
            n_keypoint_descriptor = sum([param.nelement() for param in self.local_descriptor_decoder.parameters()])
            n_saliency_regressor = sum([param.nelement() for param in self.local_sigma_regressor.parameters()])
            print(f'kp. head: {n_local_head/1000:.1f}   kp. regressor {n_keypoint_regressor/1000:.1f}   '
                  f'kp. descriptor {n_keypoint_descriptor / 1000:.1f}   '
                  f'[k] kp. saliency regresor {n_saliency_regressor/1000:.1f} [k]')
            print(f'# channels in the local map: {self.local_head.out_channels}   keypoint descriptor size: {self.local_descriptor_size}')

        n_global_head = sum([param.nelement() for param in self.global_head.parameters()])
        print(f'global descriptor head: {n_global_head/1000:.1f} [k]   pool method: {self.global_pool_method}')
        print(f'# channels in the global map: {self.global_channels}   global descriptor size: {self.global_descriptor_size}')
