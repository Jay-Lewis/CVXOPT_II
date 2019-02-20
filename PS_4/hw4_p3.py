import matplotlib.pyplot as plt
from descent_algos import *
import utils
from utils_ps_4 import *
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
dtype = torch.FloatTensor   #TODO: torch.cuda.FloatTensor
df = pd.read_csv("MatrixCompletion/M.csv", header=-1)
df2 = pd.read_csv("MatrixCompletion/O.csv", header=-1)

M = torch.Tensor(df.values).type(dtype)
O = torch.Tensor(df2.values).type(dtype)
M_samp = M*O

print('RANK:', np.linalg.matrix_rank(utils.to_numpy(M)))

print('=============  Data Loaded ==============')

# ---------------------------
# Run Descent Algos on Low-Rank Completion Prob
# ---------------------------

# Set up Descent Structure
T = int(1e3)
# alpha, beta = utils.get_alpha_beta(M)
beta = 1.0
data = utils.get_args_dict(('O', 'M_samp'),
                           (O, M_samp))
loss_fn = utils.nuclear_norm
project_fn = utils.proj_matrix_sample
params = utils.get_args_dict(('beta', 'T', 'dtype', 'project_fn'), (beta, T, dtype, project_fn))
gd = descent_structure(data, params, loss_fn)

# ---- Low-Rank Completion PGD (eta = 1.0) -------------
print('Run 1')
X_start = Variable(torch.zeros(np.shape(M)).type(dtype), requires_grad=True)
subgrad_fn = gd.get_torch_subgrad
step_fn = None
norm_fn = utils.matrix_rank
X_gd, error_gd_1, ranks, Xs_gd = gd.descent(project_gradient_update, subgrad_fn, X_start, norm_fn, step_fn)
test_errors_1 = p3_error(Xs_gd, M, data, params)

# ---- Low-Rank Completion PGD (eta = 1/t) -------------
print('Run 2')
X_start = Variable(torch.zeros(np.shape(M)).type(dtype), requires_grad=True)
step_fn = utils.one_by_sqrt
subgrad_fn = gd.get_torch_subgrad
norm_fn = utils.matrix_rank
X_gd, error_gd_2, ranks, Xs_gd = gd.descent(project_gradient_update, subgrad_fn, X_start, norm_fn, step_fn)
test_errors_2 = p3_error(Xs_gd, M, data, params)

# ---- Low-Rank Completion PGD (eta = 1/sqrt(t)) -------------
print('Run 3')
X_start = Variable(torch.zeros(np.shape(M)).type(dtype), requires_grad=True)
step_fn = utils.one_by_sqrt
subgrad_fn = gd.get_torch_subgrad
norm_fn = utils.matrix_rank
X_gd, error_gd_3, ranks, Xs_gd = gd.descent(project_gradient_update, subgrad_fn, X_start, norm_fn, step_fn)
test_errors_3 = p3_error(Xs_gd, M, data, params)


# ---------------------------
# Plots + Save
# ---------------------------

plt.clf()
fig, ax = plt.subplots()
plt.plot(error_gd_1, label='PGD 1.0')
plt.plot(error_gd_2, label='PGD 1/t')
plt.plot(error_gd_3, label='PGD 1/sqrt(t)')
plt.title('Training Error')
plt.legend()
plt.ylabel('Nuclear Norm (X)')
plt.savefig('p3/train_error.pdf')

plt.clf()
fig, ax = plt.subplots()
plt.plot(test_errors_1, label='PGD 1.0')
plt.plot(test_errors_2, label='PGD 1/t')
plt.plot(test_errors_3, label='PGD 1/sqrt(t)')
plt.title('Test Error')
plt.ylabel('Frobenius Norm (X-M)')
plt.legend()
plt.savefig('p3/test_error.pdf')

