import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Arial' # 'Helvetica', 'Times New Roman' 
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'font.weight': "bold"})
matplotlib.rcParams.update({'axes.labelweight': "bold"})
matplotlib.rcParams.update({'axes.titleweight': "bold"})

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
thresholds = np.linspace(0, 1, 100) # thresholds of PR curve

def plot_cdf(errors, save_path, xlabel, ylabel, title):
    errors = np.sort(errors)
    y = np.arange(len(errors))/float(len(errors))
    plt.plot(errors, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_yaw_errors(yaw_errors, save_path):
    yaw_errors = np.abs(yaw_errors) % 180

    plt.hist(yaw_errors, bins=36, range=(0, 180), density=True)

    plt.title('Yaw Error Distribution')
    plt.xlabel('Yaw Error (degrees)')
    plt.ylabel('Frequency')
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_trans_yaw_errors(yaw_errors, trans_errors, save_path):
    yaw_errors = np.abs(yaw_errors) % 180

    plt.scatter(yaw_errors, trans_errors)
    plt.xlabel('Yaw Error')
    plt.ylabel('Trans Error')
    plt.title('Trans Error vs. Yaw Error')
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()
    plt.close()


# convert numpy to pandas with column names
def data_to_df(data, label, method):
    num = len(data)
    data_new = np.empty((num, 3), dtype=object)

    for i in range(num):
        data_new[i, 0] = data[i]
        data_new[i, 1] = label
        data_new[i, 2] = method
        
    data_df = pd.DataFrame(data_new)
    data_df.columns = ['Error', 'Threshold', 'Method']
    
    return data_df


# calculate the success rate of pose estimation
def cal_recall_pe(rot_errors, trans_errors, num_queries = None, rot_thre = 5, trans_thre = 2, dataset = 'nclt', method = 'RING#-L', return_data = False):
    assert len(rot_errors) == len(trans_errors), 'The length of rotation errors and translation errors must be the same'
    if dataset == 'nclt':
        dataset = 'NCLT'
    elif dataset == 'oxford':
        dataset = 'Oxford'
    if 'RING#' in method:
        method = f'{method} (Ours)'
    num = 0
    num_errors = len(rot_errors)
    data = np.empty((num_errors, 3), dtype=object)
    for i in range(num_errors):
        rot_err = rot_errors[i]
        trans_err = trans_errors[i]
        data[i, 0] = dataset
        data[i, 1] = method
        data[i, 2] = 0
        if rot_err <= rot_thre and trans_err <= trans_thre:
            num += 1
            data[i, 2] = 1
    if num_queries is None:
        num_queries = num_errors
    recall_pe = num / num_queries
    # print(f'PE success rate at {trans_thre} m and {rot_thre} degrees: {recall_pe}')
    if return_data:
        data_df = pd.DataFrame(data)
        data_df.columns = ['Dataset', 'Method', 'Success']
        return recall_pe, data_df
    else:
        return recall_pe
