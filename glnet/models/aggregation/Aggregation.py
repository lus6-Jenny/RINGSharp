import setlog
import torch.nn.functional as func
import torch.nn as nn
import torch
import math
import os
import collections as coll


logger = setlog.get_logger(__name__)


def select_desc(name, params):
    if name == 'RMAC':
        agg = RMAC(**params)
    elif name == 'RAAC':
        agg = RAAC(**params)
    elif name == 'RMean':
        agg = RMean(**params)
    elif name == 'SPOC':
        agg = SPOC(**params)
    elif name == 'Embedding':
        agg = Embedding(**params)
    elif name == 'NetVLAD':
        agg = NetVLAD(**params)
    elif name == 'Encoder':
        agg = nn.Sequential(
            coll.OrderedDict(
                [
                    ('feat', eval(params['base_archi'])(**params['base_archi_param'])),
                    ('agg', select_desc(params['agg'], params['agg_param']))
                ]
            )
        )

    else:
        raise ValueError('Unknown aggregation method {}'.format(name))

    return agg


class RMAC(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.R = kwargs.pop('R', 1)  # R=1, Global Max pooling
        self.norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, feature):
        x = func.adaptive_max_pool2d(feature, (self.R, self.R))
        x = x.view(x.size(0), -1)
        if self.norm:
            x = func.normalize(x)

        return x

    def get_training_layers(self, layers_to_train=None):
        return []


class RMean(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.R = kwargs.pop('R', 1)  # R=1, Global Max pooling
        self.norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, feature):
        x = func.adaptive_avg_pool2d(feature, (self.R, self.R))
        x = x.view(x.size(0), -1)
        if self.norm:
            x = func.normalize(x)

        return x


class SPOC(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, feature):
        x = torch.sum(torch.sum(feature, dim=-1), dim=-1)
        x = x.view(x.size(0), -1)
        if self.norm:
            x = func.normalize(x)

        return x


class RAAC(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.R = kwargs.pop('R', 1)  # R=1, Global Max pooling
        self.norm = kwargs.pop('norm', True)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def forward(self, feature):
        x_pos = func.adaptive_max_pool2d(feature, (self.R, self.R))
        x_all = func.adaptive_max_pool2d(torch.abs(feature), (self.R, self.R))
        mask = x_all > x_pos
        x = x_all * (-(mask == 1).float()) + x_all * (1 - (mask == 1).float())

        x = x.view(x.size(0), -1)
        if self.norm:
            x = func.normalize(x)

        return x


class Embedding(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        agg = kwargs.pop('agg', 'RMAC')
        agg_params = kwargs.pop('agg_params', {'R': 1, 'norm': True})
        input_size = kwargs.pop('input_size', 256)
        size_feat = kwargs.pop('feat_size', 256)
        self.gate = kwargs.pop('gate', False)
        self.res = kwargs.pop('res', False)
        kernel_size = kwargs.pop('kernel_size', 1)
        stride = kwargs.pop('stride', 1)
        padding = kwargs.pop('padding', 0)
        dilation = kwargs.pop('dilation', 1)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.embed = nn.Conv2d(input_size, size_feat,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.descriptor = select_desc(agg, agg_params)

        if self.gate:
            self.gatenet = nn.Sequential(
                nn.Conv2d(input_size, size_feat, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, feature):
        embeded_feature = self.embed(feature)
        if self.res:
            embeded_feature += feature
        if self.gate:
            gating = self.gatenet(embeded_feature)
            desc = self.descriptor(embeded_feature*gating)
        else:
            desc = self.descriptor(embeded_feature)

        return desc


class NetVLADPCA(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        vlad_param = kwargs.pop('vlad_param', dict())
        pca_input_size = kwargs.pop('pca_input_size', 16384)
        pca_output_size = kwargs.pop('pca_output_size', 256)
        load = kwargs.pop('load', dict())

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        self.vlad = NetVLAD(**vlad_param)
        self.pca_fc = nn.Linear(pca_input_size, pca_output_size, bias=False)

        if load is not None:
            #clusters = torch.load(os.environ['CNN_WEIGHTS'] + load)
            pca_param = torch.load(load)
            self.pca_fc.weight = pca_param

    def forward(self, x):
        vlad = self.vlad(x)
        pca = self.pca_fc(vlad)
        if self.norm:
            pca = func.normalize(pca)

        return pca

    def get_training_layers(self, layers_to_train=None):
        return [{'params': self.vlad.parameters()}, {'params': self.pca_fc.parameters()},]


class NetVLAD(nn.Module):
    """
        Code from Antoine Miech
        @ https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/loupe.py
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_size = kwargs.pop('feature_size', None)
        self.cluster_size = kwargs.pop('cluster_size', None)
        add_batch_norm = kwargs.pop('add_batch_norm', False)
        load = kwargs.pop('load', None)
        alpha = kwargs.pop('alpha', 50)
        trace = kwargs.pop('trace', False)
        self.feat_norm = kwargs.pop('feat_norm', True)
        self.add_bias = kwargs.pop('bias', False)
        one_d_bias = kwargs.pop('one_d_bias', False)
        self.layers_to_train = kwargs.pop('layers_to_train', 'no_layer')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        # Reweighting
        self.clusters = nn.Parameter((1 / math.sqrt(self.feature_size))
                                     * torch.randn(self.feature_size, self.cluster_size))

        # Bias
        if self.add_bias:
            if one_d_bias:
                self.bias = nn.Parameter(
                    (
                        (1 / math.sqrt(self.feature_size)) * torch.randn(self.cluster_size)
                    ).view(self.cluster_size)
                )
            else:
                self.bias = nn.Parameter(
                    (
                        (1 / math.sqrt(self.feature_size)) * torch.randn(self.cluster_size)
                    ).view(1, self.cluster_size)
                )

        # Cluster
        self.clusters2 = nn.Parameter((1 / math.sqrt(self.feature_size))
                                      * torch.randn(1, self.feature_size, self.cluster_size))
        if load is not None:
            #clusters = torch.load(os.environ['CNN_WEIGHTS'] + load)
            clusters = torch.load(load)
            if self.add_bias:
                self.bias.data = -1*alpha*torch.norm(clusters, p=2, dim=1)
            self.clusters2.data = clusters
            self.clusters.data = 2*alpha*clusters.squeeze()
            logger.info('Custom clusters {} have been loaded'.format(load))
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(self.cluster_size)
        self.out_dim = self.cluster_size * self.feature_size
        self.trace = trace

    def forward(self, x):
        max_sample = x.size(2)*x.size(3)

        if self.feat_norm:
            # Descriptor-wise L2-normalization (see paper)
            x = func.normalize(x)

        x = x.view(x.size(0), self.feature_size, max_sample).transpose(1,2).contiguous()
        x = x.view(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters) + self.bias if self.add_bias else  torch.matmul(x, self.clusters)
        if self.add_batch_norm:
            assignment = self.batch_norm(assignment)

        assignment = func.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        if self.trace:
        # print(torch.max(assignment[0,0]))
            s_tmp = list()
            soft_idx = list()
            for assa in assignment:
                for assa2 in assa:
                    sorted = torch.sort(assa2, descending=True)
                    s_tmp.append(sorted[0][0]/sorted[0][1])
                    soft_idx.append(sorted[1][0])
            print(sum(s_tmp)/len(s_tmp))
            #print(soft_idx)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        # L2 intra norm
        vlad = func.normalize(vlad)

        # flattening + L2 norm
        #vlad = vlad.view(-1, self.cluster_size * self.feature_size)
        vlad = vlad.contiguous().view(-1, self.cluster_size * self.feature_size)
        vlad = func.normalize(vlad)

        return vlad

    def get_training_layers(self, layers_to_train=None):
        if not layers_to_train:
            layers_to_train = self.layers_to_train

        if layers_to_train == 'all':
            train_parameters = [{'params': self.parameters()}]
        elif layers_to_train == 'no_layer':
            train_parameters = []
        else:
            raise KeyError('No behaviour for layers_to_train = {}'.format(layers_to_train))
        return train_parameters
