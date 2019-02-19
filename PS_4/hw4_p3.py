import matplotlib.pyplot as plt
from descent_algos import *
import utils
from torch.autograd import Variable
import torch
import pandas as pd



# ==========================================
# HW 4 (Problem 3)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
print('============= Loading Data ==============')
dtype = torch.cuda.FloatTensor
df = pd.read_csv("MatrixCompletion/M.csv", header=-1)
df2 = pd.read_csv("MatrixCompletion/O.csv", header=-1)

M = torch.Tensor(df.values).type(dtype)
O = torch.Tensor(df2.values).type(dtype)
M_samp = M*O

print('=============  Data Loaded ==============')
print('M Data:')
print(M)
print(M_samp)

# ---------------------------
# Run Descent Algos on Low-Rank Completion Prob
# ---------------------------

# Set up Descent Structure
T = int(1e3)
# alpha, beta = utils.get_alpha_beta(M)
beta = None
data = utils.get_args_dict(('O', 'M_samp'),
                           (O, M_samp))
loss_fn = utils.nuclear_norm
project_fn = utils.proj_matrix_sample
params = utils.get_args_dict(('beta', 'T', 'dtype', 'project_fn'), (beta, T, dtype, project_fn))
gd = descent_structure(data, params, loss_fn)

# Low-Rank Completion PGD
X_start = Variable(torch.zeros(np.shape(M)).type(dtype), requires_grad=True)
print('X_start:')
print(X_start)
print(utils.frobenius_norm(X_start-M, None, None)**2)
subgrad_fn = gd.get_torch_subgrad
X_gd, error_gd, _, Xs_gd = gd.descent(project_gradient_update, subgrad_fn, None, X_start)
diffs = [X_i - M for X_i in Xs_gd]
errors = utils.test_error(diffs, utils.frobenius_norm, data, params)
errors = [utils.to_numpy(error)**2 for error in errors]

print('final_prediction:')
print(X_gd)
print(utils.frobenius_norm(X_gd-M, None, None)**2)
print(X_gd-M)
# ---------------------------
# Plots + Save
# ---------------------------

plt.clf()
fig, ax = plt.subplots()
plt.plot(errors, label='Project. GD')
plt.title('Training Error')
plt.legend()
plt.savefig('p3/train_error.pdf')

