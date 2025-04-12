# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch

from superpoint import SuperPoint
from superglue import SuperGlue
from sift import SIFT
from nearest_neighbor import NearestNeighbor

class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.config = config
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))
        self.sift = SIFT(config.get('sift', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        extractor = data.get('extractor', 'superpoint')
        if 'keypoints0' not in data:
            if extractor == 'superpoint':
                pred0 = self.superpoint({'image': data['image0']})
                pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
                # print(' ---- sp ----', pred0['keypoints'][0].shape, pred0['descriptors'][0].shape, pred0['scores'][0].shape)
            elif extractor == 'sift':
                pred0 = self.sift({'image': data['image0']})
                # pred = {**pred, **{k+'0': [v.float().cuda()] for k, v in pred0.items()}}
                pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
                # print(' ---- sift ----', pred0['keypoints'][0].shape, pred0['descriptors'][0].shape, pred0['scores'][0].shape)
        if 'keypoints1' not in data:
            if extractor == 'superpoint':
                pred1 = self.superpoint({'image': data['image1']})
                pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
            elif extractor == 'sift':
                pred1 = self.sift({'image': data['image1']})
                # pred = {**pred, **{k+'1': [v.float().cuda()] for k, v in pred1.items()}}
                pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
            
        # Perform the matching
        has_none_value = any(value is None for value in pred.values())
        if has_none_value:
            pred_matcher = {
                'matches0': None,
                'matches1': None,
                'matching_scores0': None,
                'matching_scores1': None,}
        else:
            matcher = data.get('matcher', 'superglue')
            if matcher == 'superglue':
                pred = {**pred, **self.superglue(data)}
            elif 'nn' in matcher:
                nnmatcher = NearestNeighbor(self.config.get(matcher, {}))
                pred = {**pred, **nnmatcher(data)}
            
        return pred