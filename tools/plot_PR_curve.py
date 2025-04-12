import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.weight': "bold"})
matplotlib.rcParams.update({'axes.labelweight': "bold"})
matplotlib.rcParams.update({'axes.titleweight': "bold"})
import numpy as np
from sklearn.neighbors import KDTree


def calculate_dist(pose1, pose2):
    dist = np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
    return dist


def compute_PR_pairs(pair_dists, query_positions, map_positions, thresholds, save_path=None, revisit_threshold: float=10.0, seq_num=1):
    tree = KDTree(map_positions)
    num_thresholds = len(thresholds)
    num_hits = np.zeros(num_thresholds) 
    num_false_alarms = np.zeros(num_thresholds) 
    num_correct_rejections = np.zeros(num_thresholds) 
    num_misses = np.zeros(num_thresholds)
    
    # import pdb; pdb.set_trace()
    for i in range(seq_num-1, pair_dists.shape[0]):
        dist = pair_dists[i+1-seq_num:i+1]
        if seq_num == 1:
            min_dist = np.min(dist)
            idx_top1 = np.argmin(dist)
        else:
            min_dist = np.max([np.min(dist[k]) for k in range(seq_num)])
            idx_top1 = np.argmin(dist[-1])
        query_position = query_positions[i]
        indices = tree.query_radius(query_position.reshape(1,-1), revisit_threshold)[0]
        for j, thre in enumerate(thresholds):
            if(min_dist < thre):
                # if under the theshold, it is considered seen.
                # and then check the correctness
                real_dist = calculate_dist(query_position, map_positions[idx_top1])
                if real_dist < revisit_threshold:
                    # TP: Hit
                    num_hits[j] = num_hits[j] + 1
                else:
                    # FP: False Alarm 
                    num_false_alarms[j] = num_false_alarms[j] + 1
            
            else:
                if len(indices) == 0:
                    # TN: Correct rejection
                    num_correct_rejections[j] = num_correct_rejections[j] + 1
                else:           
                    # FN: MISS
                    num_misses[j] = num_misses[j] + 1 
    
    precisions = num_hits / (num_hits + num_false_alarms + 1e-10)
    recalls = num_hits / (num_hits + num_misses + 1e-10)
    precisions[np.isnan(precisions)] = 1.0
    recalls[np.isnan(recalls)] = 0.0
    precisions[num_hits == 0] = 1.0 # start from (0, 1)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    if save_path is not None:
        # plot PR curve
        marker = 'x'
        markevery = 0.03
        plt.clf()
        plt.rcParams.update({'font.size': 16})
        fig = plt.figure()
        plt.plot(recalls, precisions, label=f'{revisit_threshold}m', marker=marker, markevery=markevery)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("Recall [%]")
        plt.ylabel("Precision [%]")
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
        # plt.show()
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    return precisions, recalls, f1s


def compute_AP(precisions, recalls):
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i-1])*precisions[i]
    return ap


def compute_AUC(precisions, recalls):
    auc = 0.0
    for i in range(len(precisions) - 1):
        auc += (recalls[i+1] - recalls[i])*(precisions[i+1] + precisions[i]) / 2.0
    return auc
