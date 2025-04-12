# Zhejiang University

import argparse
import numpy as np
import os
import cv2
import time
import tqdm
import torch
import random
from typing import List
from glnet.utils.params import TrainingParams, ModelParams
from glnet.datasets.base_datasets import EvaluationTuple, EvaluationSet, get_pointcloud_loader, get_pointcloud_with_image_loader


class Evaluator:
    def __init__(self, dataset_root: str, dataset_type: str, eval_set_pickle: str, device: str, params: ModelParams,
                 radius: List[float] = [1.5, 5, 20], k: int = 50, n_samples: int = None, debug: bool = False):
        # radius: list of thresholds (in meters) to consider an element from the map sequence a true positive
        # k: maximum number of nearest neighbours to consider
        # n_samples: number of samples taken from a query sequence (None=all query elements)

        assert os.path.exists(dataset_root), f"Cannot access dataset root: {dataset_root}"
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type
        self.eval_set_filepath = os.path.join(dataset_root, eval_set_pickle)
        self.device = device
        self.radius = radius
        self.k = k
        self.n_samples = n_samples
        self.debug = debug
        self.params = params

        assert os.path.exists(self.eval_set_filepath), f'Cannot access evaluation set pickle: {self.eval_set_filepath}'
        self.eval_set = EvaluationSet()
        self.eval_set.load(self.eval_set_filepath)
        if debug:
            # Make the same map set and query set in debug mdoe
            self.eval_set.map_set = self.eval_set.map_set[:20]
            self.eval_set.query_set = self.eval_set.map_set[:20]

        if n_samples is None or len(self.eval_set.query_set) <= n_samples:
            self.n_samples = len(self.eval_set.query_set)
        else:
            self.n_samples = n_samples

        if self.params.use_rgb:
            self.pcim_loader = get_pointcloud_with_image_loader(self.dataset_type)
        else:
            self.pcim_loader = get_pointcloud_loader(self.dataset_type)

    def evaluate(self, model, *args, **kwargs):
        map_embeddings = self.compute_embeddings(self.eval_set.map_set, model)
        query_embeddings = self.compute_embeddings(self.eval_set.query_set, model)

        map_positions = self.eval_set.get_map_positions()
        query_positions = self.eval_set.get_query_positions()

        # Dictionary to store the number of true positives for different radius and NN number
        tp = {r: [0] * self.k for r in self.radius}
        query_indexes = random.sample(range(len(query_embeddings)), self.n_samples)

        # Randomly sample n_samples clouds from the query sequence and NN search in the target sequence
        for query_ndx in tqdm.tqdm(query_indexes):
            # Check if the query element has a true match within each radius
            query_pos = query_positions[query_ndx]

            # Nearest neighbour search in the embedding space
            query_embedding = query_embeddings[query_ndx]
            embed_dist = np.linalg.norm(map_embeddings - query_embedding, axis=1)
            nn_ndx = np.argsort(embed_dist)[:self.k]

            # Euclidean distance between the query and nn
            delta = query_pos - map_positions[nn_ndx]  # (k, 2) array
            euclid_dist = np.linalg.norm(delta, axis=1)  # (k,) array
            # Count true positives for different radius and NN number
            tp = {r: [tp[r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(self.k)] for r in
                  self.radius}

        recall = {r: [tp[r][nn] / self.n_samples for nn in range(self.k)] for r in self.radius}
        # percentage of 'positive' queries (with at least one match in the map sequence within given radius)
        return {'recall': recall}

    def compute_embedding(self, pc, depth, imgs, model, *args, **kwargs):
        # This method must be implemented in inheriting classes
        # Must return embedding as a numpy vector
        raise NotImplementedError('Not implemented')

    def model2eval(self, model):
        # This method may be overloaded when model is a tuple consisting of a few models (as in Disco)
        model.eval()

    def compute_embeddings(self, eval_subset: List[EvaluationTuple], model, *args, **kwargs):
        self.model2eval(model)

        embeddings = None
        for ndx, e in tqdm.tqdm(enumerate(eval_subset)):
            scan_filepath = os.path.join(self.dataset_root, e.rel_scan_filepath)
            assert os.path.exists(scan_filepath)
            pc, imgs = self.pcim_loader(scan_filepath)
            pc = torch.tensor(pc)

            embedding = self.compute_embedding(pc, imgs, model)
            if embeddings is None:
                embeddings = np.zeros((len(eval_subset), embedding.shape[1]), dtype=embedding.dtype)
            embeddings[ndx] = embedding

        return embeddings
    
    def save_image(self, save_path, image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy() 
        image = (image-np.min(image))/(np.max(image)-np.min(image))*255
        image = image.astype(np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, image)
    
    def export_eval_stats(self, file_name, stats):
        with open(file_name, "a") as f:
            f.write('-------- Start --------')
            f.write('\n')
            t = time.strftime("%Y%m%d_%H%M")
            f.write(f'time: {t}')
            f.write('\n')
            for key in stats.keys():
                f.write(key + ': ')
                val = stats[key]
                if isinstance(val, dict):
                    for k in val.keys():
                        f.write(str(k) + ': ')
                        f.write(str(val[k]))
                        f.write('\n')
                else:
                    f.write(str(val))
                    f.write('\n')                    
            f.write('-------- End --------')
            f.write('\n')

    def print_results(self, global_metrics):
        for key in global_metrics.keys():
            print(key + ': ')
            val = global_metrics[key]
            if isinstance(val, dict):
                for k in val.keys():
                    print(k + ': ')
                    print(str(val[k]), end='')   
                    print('\n')
            else:
                print(str(val), end='')
                print('\n') 
    