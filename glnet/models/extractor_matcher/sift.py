import cv2
import torch
import numpy as np

from glnet.models.extractor_matcher.base_model import BaseModel

sift = cv2.SIFT_create()

def extract_SIFT_keypoints_and_descriptors(gray_img, trim_edges):
    gray_img = gray_img.astype('uint8')
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)
    kp = cv2.KeyPoint_convert(kp)

    # Trim the keypoints near to image edges
    if len(kp) != 0:
        w = gray_img.shape[1]
        valid = (kp[:, 0] >= trim_edges[0]) & (kp[:, 0] <= w - trim_edges[1])
        kp = kp[valid]
        desc = desc[valid]

    return kp, desc


class SIFT(BaseModel):
    default_conf = {
        'grayscale': True,
        'resize_max': 1600,
        'trim_edges': [0, 0],
    }
    required_inputs = ['image']    
    
    def _init(self, conf):
        pass 
    
    def _forward(self, data):
        grayscale = self.conf['grayscale']
        trim_edges = self.conf['trim_edges']
        image = data['image']
        
        if not grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = extract_SIFT_keypoints_and_descriptors(image, trim_edges)
        scores = np.ones(len(keypoints))
        
        if descriptors is None:
            return {
                'keypoints': None,
                'descriptors': None,
                'scores': None,
            }
        else:
            return {
                'keypoints': torch.tensor(keypoints)[None],
                'descriptors': torch.tensor(descriptors).t()[None],
                'scores': torch.tensor(scores)[None],
            }
