import sys
# Template optimization requires CPLEX environment, append your own cplex path as follows
sys.path.append('/usr/pkg/cplex-studio-12.8/cplex/python/3.6/x86-64_linux/')

import cplex
import scipy.io as sio
from cplex.exceptions import CplexError
import numpy as np
from matplotlib import pyplot as plt
from utils import bg_existence_labels, vector2symmatrix

def setmip(y, M, s, prec):
    """
    y: true label 
    M: match matrix (n*m). M_{ij} = 1 means design i matches prototype j
    s: sparsity threshold
    prec: precision threshold
    """
    
    n, m = M.shape
#    P = sum(y) # number of postiive samples
    
    obj_coef = [1.0]*n + [0.0]*m 
    
    var_ub = [1.0]*(n+m)
    var_lb = [0.0]*(n+m)
    var_type = "I"*(n+m)
    var_names = ["x{0}".format(i) for i in range(n+m)]
    
    rhs = [0.0]*n + [0.0]*n + [s] + [0.0]
    sense = "G"*n + "L"*n + "L" + "G"
    
    ell = list(range(n))
    c = list(range(n, n+m))
    y = y.tolist()
    M = M.tolist()
    cst = [[c+[i],M[i]+[-1]] for i in range(n)] + [[c+[i],M[i]+[-m]] for i in range(n)] + [[c, [1]*m]] + [[ell, [yi-prec for yi in y]]]
    cst_names = ["cst_prediction_ge_" + str(i) for i in range(n)] +\
                ["cst_prediction_le_" + str(i) for i in range(n)] +\
                ["sparsity"] + ["precision"]
    return obj_coef, var_ub, var_lb, var_type, var_names, rhs, sense, \
        cst, cst_names

def solvemip(obj_coef, var_ub, var_lb, var_type, var_names, rhs, sense, \
        cst, cst_names, timelim=60):
    
    try:
        model = cplex.Cplex()
        model.parameters.timelimit.set(timelim)
        model.parameters.emphasis.memory.set(True)
        model.parameters.mip.tolerances.mipgap.set(1e-5)
        model.parameters.emphasis.mip.set(1)
        model.parameters.advance.set(1)
        model.parameters.mip.strategy.heuristicfreq.set(10)
        model.parameters.mip.strategy.nodeselect.set(2)
        model.parameters.mip.tolerances.integrality.set(0)
        
        # print out
        model.parameters.mip.display.set(4)
        
        
        model.objective.set_sense(model.objective.sense.maximize)
        model.variables.add(obj=obj_coef, lb=var_lb, ub=var_ub, 
                            types=var_type, names=var_names)
        model.linear_constraints.add(lin_expr=cst, senses=sense, 
                                     rhs=rhs, names=cst_names)
        
        model.solve()
    except CplexError as exc:
        print(exc)
        
    print("Solution status = ", model.solution.get_status(), ":", end=' ')
    print(model.solution.status[model.solution.get_status()])
    print("Solution value  = ", model.solution.get_objective_value())
    print('Optimality gap = ', model.solution.MIP.get_mip_relative_gap())
    print('absolute lb = ', model.solution.MIP.get_best_objective())
    return model

def get_param(model, n):
    var = model.solution.get_values()
    yhat = np.array(var[:n])
    c = np.array(var[n:])
    return yhat, c

def plot_templates(templates):
    """
    plot unit-cell templates
    """
    templates_mat = vector2symmatrix(templates)
    for i, template in enumerate(templates_mat):
        plt.figure()
        template[np.where(template<0)] = 0.5
        plt.imshow(template,vmin=0,vmax=1)
        plt.axis('off')
        plt.savefig(f'./plots/template_{freq_ranges[bg,0]}_{freq_ranges[bg,1]}_{i}.jpg')

# load the data
data = sio.loadmat('./data/bandgap_data.mat')
dispersion = data['dispersion'] # dispersion curves
freq_ranges = np.array([[0,600],[600,1200],[1200,1800],[1800,2400],[2400,3000]]) # target ranges
labels = bg_existence_labels(dispersion, freq_ranges)

bg = 0 # band gap index
y = labels[:,bg]
precision_threshold = 0.95

valid_designs = np.load(f'./results/valid_designs_{freq_ranges[bg,0]}_{freq_ranges[bg,1]}_{int(precision_threshold*100)}.npy')
match_matrix = np.load(f'./results/match_mat_{freq_ranges[bg,0]}_{freq_ranges[bg,1]}_{int(precision_threshold*100)}.npy').transpose((1,0))

# solving MIP
print('start setting MIP')
obj_coef, var_ub, var_lb, var_type, var_names, rhs, sense, \
        cst, cst_names = setmip(y, match_matrix, 5, 0.99)
print('start solving MIP')
model = solvemip(obj_coef, var_ub, var_lb, var_type, var_names, rhs, sense, \
        cst, cst_names, timelim=3600)

# MIP results
y_hat, c = get_param(model, match_matrix.shape[0])
print('support:',y_hat.sum())
print('prototype chosen:',np.where(c==1))
print('precision:', (y_hat * y).sum()/y_hat.sum())
print('freq range:', freq_ranges[bg])

# get chosen templates
chosen_designs = valid_designs[c==1]
plot_templates(chosen_designs)