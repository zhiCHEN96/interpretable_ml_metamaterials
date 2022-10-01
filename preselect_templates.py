import numpy as np
from itertools import combinations
import time
import scipy.io as sio
from utils import bg_existence_labels

# load the data
data = sio.loadmat('./data/bandgap_data.mat')
x = data['feature_raw'] # raw features
dispersion = data['dispersion'] # dispersion curves

freq_ranges = np.array([[0,600],[600,1200],[1200,1800],[1800,2400],[2400,3000]]) # target ranges
labels = bg_existence_labels(dispersion, freq_ranges)

idx_all = np.arange(15)
precision_threshold = 0.95

for bg in range(freq_ranges.shape[0]):
    start = time.time()
    y = labels[:,bg].astype('int8')
    valid_designs = []
    precisions = []
    supports = []
    match_mat = []

    for n_free in range(6,14): # number of free pixels
        n_fix = 15 - n_free # number of fixed pixels
        x_fix = x[:1<<n_fix,:n_fix]
        x_free = x[:1<<n_free,:n_free]
        support = 1<<n_free
        for idx in combinations(idx_all,n_fix): # try all combinations
            idx_fix = np.array(idx)
            idx_free = np.setdiff1d(idx_all,idx_fix)
            sum_fix = (x_fix<<idx_fix).sum(1)
            sum_free = (x_free<<idx_free).sum(1)
            idx_match = np.tile(sum_free,(1<<n_fix,1))+sum_fix.reshape((-1,1))
            for i in range(1<<n_fix):
                precision = y[idx_match[i]].mean()
                if precision > precision_threshold:
                    design = -np.ones(15,dtype='int8')
                    design[idx_fix] = x_fix[i]
                    valid_designs.append(design)
                    precisions.append(precision)
                    supports.append(support)
                    match_row = np.zeros(y.shape[0],dtype='int8')
                    match_row[idx_match[i]] = 1
                    match_mat.append(match_row)
                    
    end = time.time()
    print('total time:', end-start)

    valid_designs = np.array(valid_designs)
    precisions = np.array(precisions)
    supports = np.array(supports)
    match_mat = np.array(match_mat)

    print(match_mat.shape)
    np.save(f'./results/valid_designs_{freq_ranges[bg,0]}_{freq_ranges[bg,1]}_{int(precision_threshold*100)}.npy',valid_designs)
    np.save(f'./results/match_mat_{freq_ranges[bg,0]}_{freq_ranges[bg,1]}_{int(precision_threshold*100)}.npy',match_mat)
