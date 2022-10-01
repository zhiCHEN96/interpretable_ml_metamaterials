import numpy as np


def bg_existence_labels(dispersion, freq_ranges, size_min = 0.01):
    """
    Calculate whether a band gap exist in certain freq range
    
    Inputs:
    dispersion: dispersion curves
    freq_range: K*2, K target frequency ranges
    size_min: minimum size of the bandgap, used to cut nonrobust band gaps
    
    Output:
    labels: n*K, column j represents band gap existence of the jth frequency range
    """
    freq_lower = dispersion[:,:-1,:].max(2) # lower frequency of the gap
    freq_upper = dispersion[:,1:,:].min(2) # upper frequency of the gap
    
    labels = np.zeros((dispersion.shape[0],freq_ranges.shape[0]))
    for i in range(freq_ranges.shape[0]):
        labels[:,i] = ((np.fmin(freq_upper,freq_ranges[i,1])-np.fmax(freq_lower,freq_ranges[i,0])>size_min).astype('int32').sum(1)>0).astype('int32')
    
    return labels

def vector2symmatrix(x):
    """
    Get the symmetric unit cell matrix based on the unique pixels (raw features).
    Please see Figure 2a in our paper for the details of the construction
    Input:
        x: unique pixels, suppose the unitcell is d*d, x is n * (d/2*(d/2+1)/2)
    Output:
        x_mat: d*d unit cell matrix constructed from unique pixels
    """
    n, n_unique = x.shape[0], x.shape[1]
    d = int(np.sqrt(n_unique*8+1)-1)
    x_mat = np.zeros((n,d,d))
    
    for index in range(n_unique):
        sumOfInc = 1
        realIndex = n_unique - (index - 1)
        row = d//2 - 1
        col = d//2 - 1
        
        for y in range(1, d//2+1):
            sumOfInc += y
            if sumOfInc >= realIndex:
                col = y - 1
                row = realIndex - (sumOfInc - y) - 1
                x_mat[:,row,col] = x[:,index]
                x_mat[:,d-1-row, col] = x[:,index]
                x_mat[:,row, d-1-col] = x[:,index]
                x_mat[:,d-1-row, d-1-col] = x[:,index]
                x_mat[:,col,row] = x[:,index]
                x_mat[:,d-1-col, row] = x[:,index]
                x_mat[:,col, d-1-row] = x[:,index]
                x_mat[:,d-1-col, d-1-row] = x[:,index]                
                break
            
    return x_mat

def shape_frequency_features(x, windows):
    """
    Get the shape frequency features from raw features
    Input:
        x: n*n_unique, raw features
        windows: tuple of k sliding windows (shapes)
    Output:
        x_sff: n*k shape frequency features
    """
    
    x_mat = vector2symmatrix(x) # get the entire unitcell
    d = x_mat.shape[1]
    x_mat_rep = np.tile(x_mat, (1,2,2)) # repeat 2 times to calculate SFF easily
    x_sff = np.zeros((x.shape[0], len(windows)))
    
    for i, window in enumerate(windows):
        w1, w2 = window.shape[0], window.shape[1] # window shape
        for j in range(d):
            for k in range(d):
                # only calculate exact matching between the window and soft pixels
                x_sff[:,i] += ((x_mat_rep[:,j:j+w1,k:k+w2] * window).sum((1,2)) == 0)
    x_sff /= d*d # normalize to 0.0-1.0
    
    return x_sff
    
