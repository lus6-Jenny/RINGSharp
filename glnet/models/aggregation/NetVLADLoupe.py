"""
NetVLAD Pytorch implementation.
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math


class NetVLADLoupe(nn.Module):
    """
    Original Tensorflow implementation: https://github.com/antoine77340/LOUPE
    """
    def __init__(self, feature_size, cluster_size, output_dim,
                 gating=True, add_norm=True, is_training=True, normalization='batch'):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_norm
        self.cluster_size = cluster_size
        if normalization == 'instance':
            norm = lambda x: nn.LayerNorm(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(8, x)
        else:
            norm = lambda x: nn.BatchNorm1d(x)
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_norm:
            self.cluster_biases = None
            self.bn1 = norm(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = norm(output_dim)

        if gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_norm, normalization=normalization)

    def forward(self, x):
        # x = x.transpose(1, 3).contiguous()
        batch_size = x.shape[0]
        feature_size = x.shape[-1]
        x = x.view((batch_size, -1, feature_size))
        max_samples = x.shape[1]
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1).contiguous()
        vlad0 = vlad - a

        vlad1 = F.normalize(vlad0, dim=1, p=2, eps=1e-6)
        vlad2 = vlad1.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad2, dim=1, p=2, eps=1e-6)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    """
    Original Tensorflow implementation: https://github.com/antoine77340/LOUPE
    """
    def __init__(self, dim, add_batch_norm=True, normalization='batch'):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        if normalization == 'instance':
            norm = lambda x: nn.LayerNorm(x)
        elif normalization == 'group':
            norm = lambda x: nn.GroupNorm(8, x)
        else:
            norm = lambda x: nn.BatchNorm1d(x)
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = norm(dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


# from https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLADLayer(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLADLayer, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


if __name__ == '__main__':
    net_vlad = NetVLADLoupe(feature_size=1024, max_samples=360, cluster_size=16,
                                 output_dim=20, gating=True, add_batch_norm=True,
                                 is_training=True)
    # input  (bs, 1024, 360, 1)
    torch.manual_seed(1234)
    input_tensor = F.normalize(torch.randn((1,1024,360,1)), dim=1)
    input_tensor2 = torch.zeros_like(input_tensor)
    input_tensor2[:, :, 2:, :] = input_tensor[:, :, 0:-2, :].clone()
    input_tensor2[:, :, :2, :]  = input_tensor[:, :, -2:, :].clone()
    input_tensor2= F.normalize(input_tensor2, dim=1)
    input_tensor_com = torch.cat((input_tensor, input_tensor2), dim=0)

    # print(input_tensor[0,0,:,0])
    # print(input_tensor2[0,0,:,0])
    print("==================================")

    with torch.no_grad():
        net_vlad.eval()
        # output_tensor = net_vlad(input_tensor_com)
        # print(output_tensor)
        out1 = net_vlad(input_tensor)
        print(out1)
        net_vlad.eval()
        # input_tensor2[:, :, 20:, :] = 0.1
        input_tensor2 = F.normalize(input_tensor2, dim=1)
        out2 = net_vlad(input_tensor2)
        print(out2)
        net_vlad.eval()
        input_tensor3 = torch.randn((1,1024,360,1))
        out3 = net_vlad(input_tensor3)
        print(out3)
        print(((out1-out2)**2).sum(1))
        print(((out1-out3)**2).sum(1))