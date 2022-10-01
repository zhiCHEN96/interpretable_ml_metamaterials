import pandas
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import bg_existence_labels, shape_frequency_features

def binarize(df):
    """binarize continous feature"""
    colnames = df.columns
#    print(colnames)
    for i in range(len(colnames)-1):
        print(i)
        uni = df[colnames[i]].unique()
        uni.sort()
        for j in range(len(uni)-1):
            df[colnames[i]+'<'+str(uni[j])] = 1
            k = df[colnames[i]] >= uni[j]
            df[colnames[i]+'<'+str(uni[j])][k] = 0 
        df = df.drop(colnames[i], axis=1)
    df['bg'] = df[colnames[len(colnames)-1]].astype('int64')
    df = df.drop(colnames[len(colnames)-1], axis=1)
    return df

def create_dataset(x_sff, labels, file_name):
    """
    Create dataset with binary feature for GOSDT
    Inputs:
    x_sff: shape frequency features
    labels: band gap labels
    file_name: suffix of file name    
    """
    
    folder = './data/'
    for bg in range(labels.shape[1]):
        y = labels[:,bg].astype('int32')
        
        df = pd.DataFrame(np.hstack((x_sff,y.reshape((-1,1)))))
        df.columns = ['x' + str(i) for i in range(x_sff.shape[1])] + ['y']
        df = binarize(df)
        
        df_train, df_test = train_test_split(df, test_size=0.2)
        
        df_train.to_csv(folder + 'train_' + file_name + '_' + str(bg) + '.csv', index=False)
        df_test.to_csv(folder + 'test_' + file_name + '_' + str(bg) + '.csv', index=False)

#sliding windows used
windows = (np.ones((1,1)),
           np.ones((1,2)),
           np.array([[1,1],[1,0]]),
           np.ones((2,2)),
           np.ones((1,3)),
           np.array([[1,0,1],[0,0,0],[1,0,1]]),
           np.array([[1,1,1],[1,0,1],[1,1,1]]),
           np.array([[0,1,0],[1,1,1],[0,1,0]]),
           np.array([[1,1,1],[0,0,0],[1,1,1]]),
           np.eye(3),
           np.ones((3,3)),
           np.ones((1,4)),
           np.array([[1,0,0,1]]),
           np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]),
           np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]),
           np.array([[1,1,1,1],[1,0,0,1]]),
           np.array([[0,1,1,0],[1,1,1,1],[1,1,1,1],[0,1,1,0]]),
           np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[1,1,1,1]]),
           np.eye(4),
           np.ones((2,4)),
           np.ones((4,4)),
           )
# load the data
data = sio.loadmat('./data/bandgap_data.mat')
x = data['feature_raw'] # raw features
dispersion = data['dispersion'] # dispersion curves

freq_ranges = np.array([[0,600],[600,1200],[1200,1800],[1800,2400],[2400,3000]]) # target ranges
labels = bg_existence_labels(dispersion, freq_ranges)

#sliding windows/shapes used in the paper
windows = (np.ones((1,1)), #0
           np.ones((1,2)), #1
           np.array([[1,1],[1,0]]), #2
           np.ones((2,2)), #3
           np.ones((1,3)), #4
           np.array([[1,0,1],[0,0,0],[1,0,1]]), #5
           np.array([[1,1,1],[1,0,1],[1,1,1]]), #6
           np.array([[0,1,0],[1,1,1],[0,1,0]]), #7
           np.array([[1,1,1],[0,0,0],[1,1,1]]), #8
           np.eye(3), #9
           np.ones((3,3)), #11
           np.ones((1,4)), #12
           np.array([[1,0,0,1]]), #13
           np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]), #14
           np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]]), #15
           np.array([[1,1,1,1],[1,0,0,1]]), #16
           np.array([[0,1,1,0],[1,1,1,1],[1,1,1,1],[0,1,1,0]]), #17
           np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[1,1,1,1]]), #18
           np.eye(4), #19
           np.ones((2,4)), #20
           np.ones((4,4)), #21
           )
x_sff = shape_frequency_features(x,windows)

create_dataset(x_sff, labels, 'sff')