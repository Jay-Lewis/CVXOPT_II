import matplotlib.pyplot as plt
from descent_algos import *
import utils
from utils_ps_4 import *
from torch.autograd import Variable
import torch
import pandas as pd



# ==========================================
# HW 4 (Problem 4)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
print('============= Loading Data ==============')
nrows = 1198  # 1198
dtype = torch.FloatTensor  #TODO: torch.cuda.FloatTensor
dtype2 = torch.LongTensor  #TODO: torch.cuda.LongTensor
df = pd.read_csv("digits/X_digits_train.csv", nrows=nrows, header=-1)
df2 = pd.read_csv("digits/y_digits_train.csv",nrows=nrows, header=-1)


X_train = torch.Tensor(utils.df_str_to_matrix(df)).type(dtype)
y_train = torch.Tensor(df2.values).type(dtype2)

df = pd.read_csv("digits/X_digits_test.csv",nrows=nrows, header=-1)
df2 = pd.read_csv("digits/y_digits_test.csv",nrows=nrows, header=-1)
X_test = torch.Tensor(utils.df_str_to_matrix(df)).type(dtype)
y_test = torch.Tensor(df2.values).type(dtype2)
print('=============  Data Loaded ==============')
print('X Data:')
print(np.shape(X_train))
N = np.shape(X_train)[0]

# ---------------------------
# Run Descent Algos on Logistic Regression Problem
# ---------------------------

# Set up Descent Structure
T = int(1e3)
num_c = 10; svrg_s = int(100)
beta_0 = Variable(torch.zeros(np.shape(X_train)[1], num_c).type(dtype), requires_grad=True)
alpha, beta = utils.get_alpha_beta(X_train)
data = utils.get_args_dict(('A_save', 'A', 'b_save', 'b', 'svrg_save', 'svrg_grad_save'),
                           (X_train, X_train, y_train, y_train, beta_0, beta_0.detach()))
test_data = utils.get_args_dict(('A', 'b'), (X_test, y_test))
loss_fn = utils.logist_loss
params = utils.get_args_dict(('beta', 'lam', 'T', 'c', 'dtype', 'sgd_num', 'svrg_s'),
                             (100*beta, 0.1, T, 1e-5, dtype, 1, svrg_s))
gd = descent_structure(data, params, loss_fn)

# Regression via GD
beta_start = Variable(torch.zeros(np.shape(X_train)[1], num_c).type(dtype), requires_grad=True)
subgrad_fn = gd.get_torch_subgrad
beta_gd, error_gd, _, betas_gd = gd.descent(data_stoch_sg_update, subgrad_fn, beta_start)
test_errors_gd = utils.test_error(betas_gd, loss_fn, test_data, params)


# Regression via SGD
beta_start = Variable(torch.zeros(np.shape(X_train)[1], num_c).type(dtype), requires_grad=True)
subgrad_fn = gd.get_torch_subgrad
beta_sgd, error_sgd, _, betas_sgd = gd.descent(subgradient_update, subgrad_fn, beta_start)
test_errors_sgd = utils.test_error(betas_sgd, loss_fn, test_data, params)

# Regression via SVRGD
beta_start = Variable(torch.zeros(np.shape(X_train)[1], num_c).type(dtype), requires_grad=True)
subgrad_fn = gd.get_torch_subgrad
beta_svrgd, error_svrg, _, betas_svrg = gd.descent(svrg_update, subgrad_fn, beta_start)
test_errors_svrg = utils.test_error(betas_svrg, loss_fn, test_data, params)


# ---------------------------
# Plots + Save
# ---------------------------

plt.clf()
fig, ax = plt.subplots()
xs = range(1, len(error_sgd)+1)
plt.plot(xs, error_sgd, label='Stoch GD')
xs = np.multiply(N, [elem for elem in range(1, len(error_gd)+1)])
plt.plot(xs, error_gd, label='GD')
xs = p4_get_steps(svrg_s, len(error_gd), N)
plt.plot(xs, error_svrg, label='SVRGD')
plt.title('Training Error')
ax.set_xscale("log")
plt.legend()
plt.savefig('p4/train_error.pdf')

plt.clf()
fig, ax = plt.subplots()
xs = range(1, len(error_sgd)+1)
plt.plot(xs, test_errors_sgd, label='Stoch GD')
xs = np.multiply(N, [elem for elem in range(1, len(error_gd)+1)])
plt.plot(xs, test_errors_gd, label='GD')
xs = p4_get_steps(svrg_s, len(error_gd), N)
plt.plot(xs, test_errors_svrg, label='SVRGD')
plt.title("Test Error")
ax.set_xscale("log")
plt.legend()
plt.savefig('p4/test_error.pdf')

