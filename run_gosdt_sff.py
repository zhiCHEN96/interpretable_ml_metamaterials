from gosdt.osdt_imb_v9 import bbound, predict
import pandas as pd

# please run create_bin_datasets.py first to get the binarized dataset
# load the binarized SFF dataset (both train and test)
bg = 2 # index of frequency range
df_train = pd.read_csv(f'./data/train_sff_{bg}.csv')
df_test = pd.read_csv(f'./data/test_sff_{bg}.csv')
col_names = df_train.columns

#train = df
train = df_train.values
test = df_test.values
x_train = train[:,:-1]
y_train = train[:,-1]
x_test = test[:,:-1]
y_test = test[:,-1]
lamb=0.01

# run GOSDT, it takes 10 mins
w = 1
theta = None
leaves_c, pred_c, dic_c, nleaves_c, m_c, n_c, totaltime_c, time_c, R_c, \
COUNT_c, C_c, accu_c, best_is_cart_c, clf_c = \
bbound(x_train, y_train, 'precision', lamb, prior_metric='curiosity', w=w, theta=theta, MAXDEPTH=float('Inf'), 
           MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
           support=True, incre_support=True, accu_support=False, equiv_points=True,
           lookahead=True, lenbound=True, R_c0 = 1, timelimit=600, init_cart = True,
           saveTree = False, readTree = False)

yhat, out = predict('precision', leaves_c, pred_c, nleaves_c, dic_c, x_train, y_train, best_is_cart_c, clf_c, w, theta, logon=False)
yhat, out = predict('recall', leaves_c, pred_c, nleaves_c, dic_c, x_train, y_train, best_is_cart_c, clf_c, w, theta, logon=False)
yhat, out = predict('precision', leaves_c, pred_c, nleaves_c, dic_c, x_test, y_test, best_is_cart_c, clf_c, w, theta, logon=False)
yhat, out = predict('recall', leaves_c, pred_c, nleaves_c, dic_c, x_test, y_test, best_is_cart_c, clf_c, w, theta, logon=False)

# organize the tree into readable format
for i, path in enumerate(leaves_c):
    tup = []
    for j, idx in enumerate(path):
        f_name = col_names[dic_c[abs(idx)]]
        if idx<0:
            f_name = f_name[:f_name.index('<')] + '>=' +f_name[f_name.index('<')+1:]
        tup.append(f_name)
    leaves_c[i] = tuple(tup)

# print the tree obtained by GOSDT
print("Tree:")
for i, leave in enumerate(leaves_c):
    print(f"IF {leave} PREDICT {pred_c[i]}")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# save the tree
with open(f'./results/GOSDT_tree_range_{bg}.txt','w') as f:
    f.write("Tree:\n")
    for i, leave in enumerate(leaves_c):
        f.write(f"IF {leave} PREDICT {pred_c[i]}\n")
            