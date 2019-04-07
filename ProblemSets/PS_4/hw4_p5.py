import matplotlib.pyplot as plt
from ProblemSets.descent_algos import *
from ProblemSets import utils
from torch.autograd import Variable
import torch
import pandas as pd
from scipy.io import mmread

# ==========================================
# HW 4 (Problem 5)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
print('============= Loading Data ==============')
nrows = 500  # 1198
dtype = torch.FloatTensor  #TODO: torch.cuda.FloatTensor
dtype2 = torch.LongTensor  #TODO: torch.cuda.LongTensor
df2 = pd.read_csv("news/y_news_train.csv", header=-1)
X_train = mmread("news/X_news_train.mtx")
X_train = torch.Tensor(X_train.todense()).type(dtype)
y_train = torch.Tensor(df2.values).type(dtype2)

df2 = pd.read_csv("news/y_news_test.csv", header=-1)
X_test = mmread("news/X_news_test.mtx")
X_test = torch.Tensor(X_test.todense()).type(dtype)
y_test = torch.Tensor(df2.values).type(dtype2)
print('=============  Data Loaded ==============')
print('X Data:')
print(np.shape(X_train))
N = np.shape(X_train)[0]

# ---------------------------
# Run Descent Algos on Logistic Regression Problem
# ---------------------------

# Set up Descent Structure
T = int(1e1)
num_c = 20; svrg_s = int(100)
beta_0 = Variable(torch.zeros(np.shape(X_train)[1], num_c).type(dtype), requires_grad=True)
beta_step = 1e-6
data = utils.get_args_dict(('A_save', 'A', 'b_save', 'b', 'svrg_save', 'svrg_grad_save'),
                           (X_train, X_train, y_train, y_train, beta_0, beta_0.detach()))
test_data = utils.get_args_dict(('A', 'b'), (X_test, y_test))
loss_fn = utils.logist_loss
params = utils.get_args_dict(('beta', 'lam', 'T', 'c', 'dtype', 'sgd_num', 'svrg_s'),
                             (beta_step, 0.1, T, 1e-5, dtype, 1, svrg_s))
gd = descent_structure(data, params, loss_fn)

# # Regression via SGD
# beta_start = Variable(torch.zeros(np.shape(X_train)[1], num_c).type(dtype), requires_grad=True)
# subgrad_fn = gd.get_torch_subgrad
# beta_sgd, error_sgd, _, betas_sgd = gd.descent(subgradient_update, subgrad_fn, beta_start)
# test_errors_sgd = utils.test_error(betas_sgd, loss_fn, test_data, params)

# Regression via SGD + Minibatching
params = utils.get_args_dict(('beta', 'lam', 'T', 'c', 'dtype', 'sgd_num', 'svrg_s'),
                             (beta_step, 0.1, T, 1e-5, dtype, 10, svrg_s))
gd2 = descent_structure(data, params, loss_fn)
beta_start = Variable(torch.zeros(np.shape(X_train)[1], num_c).type(dtype), requires_grad=True)
subgrad_fn = gd.get_torch_subgrad
beta_sgd, error_sgd, _, betas_sgd = gd2.descent(subgradient_update, subgrad_fn, beta_start)
test_errors_sgd = utils.test_error(betas_sgd, loss_fn, test_data, params)

# ---------------------------
# Plots + Save
# ---------------------------

plt.clf()
fig, ax = plt.subplots()
xs = range(1, len(error_sgd)+1)
plt.plot(xs, error_sgd, label='Stoch GD')
plt.title('Training Error')
ax.set_xscale("log")
plt.legend()
plt.savefig('p5/train_error.pdf')

plt.clf()
fig, ax = plt.subplots()
xs = range(1, len(error_sgd)+1)
plt.plot(xs, test_errors_sgd, label='Stoch GD')
plt.title("Test Error")
plt.legend()
plt.savefig('p5/test_error.pdf')
#
