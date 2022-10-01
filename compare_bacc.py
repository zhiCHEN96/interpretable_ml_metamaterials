import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd
import seaborn as sns
from utils import bg_existence_labels, shape_frequency_features

# load the data
data = sio.loadmat('./data/bandgap_data.mat')
x = data['feature_raw'] # raw features
dispersion = data['dispersion'] # dispersion curves

freq_ranges = np.array([[0,1000],[1000,2000],[2000,3000],[3000,4000],[4000,5000]]) # target ranges
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


T = 5 # 5 runs of the experiment
bacc = []
for t in range(T):
    print('Round', t)
    for f_type in ('raw','interpretable-sff'):    
        if f_type == 'raw':
            x_data = x
        else:
            x_data = x_sff            
        for bandgap in range(0,labels.shape[1]):
            print('freq_range',freq_ranges[bandgap,:])
            y = labels[:,bandgap].astype('int32')
            
            # train test split
            x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2)
            
            # sample weights
            w_pn_train = [y_train.sum(), (len(y_train)-y_train.sum())]
            weights_train = [w_pn_train[y] for y in y_train]
            d_train = lgb.Dataset(x_train, label = y_train, weight = weights_train)
            w_pn_test = [y_test.sum(), (len(y_test)-y_test.sum())]
            weights_test = [w_pn_test[y] for y in y_test]
            d_test = lgb.Dataset(x_test, label =  y_test, weight = weights_test)
            
            svc = SVC(kernel = 'linear',class_weight = 'balanced', probability = True)
            svc.fit(x_train, y_train)
            y_prob = svc.predict_proba(x_test)[:,1]
            y_pred = (y_prob>0.5).astype('int32')
            bacc.append(accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('SVM')
            print('bacc', accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('precision', precision_score(y_test, y_pred))
            
            lr = LogisticRegression(class_weight = 'balanced')
            lr.fit(x_train, y_train)
            y_prob = lr.predict_proba(x_test)[:,1]
            y_pred = (y_prob>0.5).astype('int32')
            bacc.append(accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('LR')
            print('bacc', accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('precision', precision_score(y_test, y_pred))
            
            rf = RandomForestClassifier(n_estimators = 300, class_weight = 'balanced',max_leaf_nodes=100)
            rf.fit(x_train, y_train)
            y_prob = rf.predict_proba(x_test)[:,1]
            y_pred = (y_prob>0.5).astype('int32')
            bacc.append(accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('RF')
            print('bacc', accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('precision', precision_score(y_test, y_pred))
            
            dt = DecisionTreeClassifier(class_weight = 'balanced',max_leaf_nodes=50)
            dt.fit(x_train, y_train)
            y_prob = dt.predict_proba(x_test)[:,1]
            y_pred = (y_prob>0.5).astype('int32')
            bacc.append(accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('CART')
            print('bacc', accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('precision', precision_score(y_test, y_pred))
            
            nn = MLPClassifier(hidden_layer_sizes=(100, 100),learning_rate='adaptive',max_iter = 400)
            nn.fit(x_train, y_train)
            y_prob = nn.predict_proba(x_test)[:,1]
            y_pred = (y_prob>0.5).astype('int32')
            bacc.append(accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('NN')
            print('bacc', accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('precision', precision_score(y_test, y_pred))
            
            param = {'num_leaves': 50, 'objective': 'binary'}
            param['metric'] = ['auc']
            bst = lgb.train(param, d_train, 300, valid_sets=None,)
            y_prob = bst.predict(x_test)
            y_pred = (y_prob>0.5).astype('int32')
            bacc.append(accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('LightGBM')
            print('bacc', accuracy_score(y_test, y_pred,sample_weight=weights_test))
            print('precision', precision_score(y_test, y_pred))

baccs = np.array(bacc).reshape((T,2,labels.shape[1],-1))
np.save('./results/baccs.npy', baccs)
baccs = np.load('./results/baccs.npy')

baccs_raw = baccs[:,0,:,:]
baccs_sff = baccs[:,1,:,:]

print('raw', baccs_raw.mean(0))
print('interpretable-sff', baccs_sff.mean(0))
print('raw', baccs_raw.std(0))
print('interpretable-sff', baccs_sff.std(0))


# baccs_cnn is the balanced accuracy of ResNet18, the numbers are obtained by running trainCNN.py for all the target ranges
baccs_cnn = np.load('./results/baccs_cnn.npy')
T = 5
baccs_cnn = baccs_cnn[:,:T]
    
# plot figures of the baccs
for i in range(T):
    bacc = baccs[:,:,i,:]
    df = pd.DataFrame()
    df['bacc'] = list(bacc.reshape((-1))) + list(baccs_cnn[i])
    df['model'] = ['SVM', 'LR', 'RF', 'CART','MLP','LightGBM']*2*T + ['CNN']*T
    df['ftype'] = (['raw']*6+['sff']*6)*T + ['raw']*T
    
    sns.set(font_scale=1.4)
    plt.figure(figsize = (7,4))
    ax = sns.barplot(x="model", y="bacc", hue="ftype", data=df)
    ax.set(ylim=(0.4, 1.02))
    plt.tight_layout()
    plt.savefig('./plots/freq_range_'+str(i)+'.jpg')
    
