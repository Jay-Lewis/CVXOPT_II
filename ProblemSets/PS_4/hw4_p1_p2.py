import matplotlib.pyplot as plt
from ProblemSets.descent_algos import *
from ProblemSets import utils
from torch.autograd import Variable
import torch


# ==========================================
# HW 4 (Problems 1 and 2)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
dtype = torch.cuda.FloatTensor
X = torch.Tensor(np.load("X.npy")).type(dtype)
y = torch.Tensor(np.load("y.npy")).type(dtype)
y_corr = torch.Tensor(np.load("y_corr.npy")).type(dtype)
#TODO: why is y = y_corr ?????
# ---------------------------
# Run Descent Algos on Robust Regression Problem
# ---------------------------

# Set up Descent Structure
T = int(1e3)
alpha, beta = utils.get_alpha_beta(X)
data = utils.get_args_dict(('A', 'b'), (X, y_corr))
test_data = utils.get_args_dict(('A', 'b'), (X, y))
loss_fn = utils.l1_loss_scaled
project_fn = utils.proj_prob_simp
params = utils.get_args_dict(('beta', 'lam', 'T', 'c', 'project_fn', 'dtype'), (1000 * beta, 10.0, T, 1e-5, project_fn, dtype))
gd = descent_structure(data, params, loss_fn)


# Regression via PGD (L1_loss)
x_start = Variable(torch.ones(np.shape(X)[1], 1).type(dtype), requires_grad=True)
norm_fn = None
subgrad_fn = gd.get_torch_subgrad
x_pgd, error_pgd, l1_pgd, xs_pgd = gd.descent(project_gradient_update_, subgrad_fn, norm_fn, x_start)
test_errors_pgd = utils.test_error(xs_pgd, utils.l2_loss_scaled, test_data, params)

print('final estimate:')
print(x_pgd)
print(torch.sum(x_pgd))

# Regression via PGD (L2 loss)
x_start_2 = Variable(torch.ones(np.shape(X)[1], 1).type(dtype), requires_grad=True)
loss_fn = utils.l2_loss_scaled
gd2 = descent_structure(data, params, loss_fn)
x_pgd_2, error_pgd_2, l1_pgd_2, xs_pgd_2 = gd2.descent(project_gradient_update, subgrad_fn, norm_fn, x_start_2)
test_errors_pgd_2 = utils.test_error(xs_pgd_2, loss_fn, test_data, params)

print('final estimate:')
print(x_pgd_2)
print(torch.sum(x_pgd_2))


# Regression via Mirror Descent
x_start = Variable(torch.ones(np.shape(X)[1], 1).type(dtype), requires_grad=True)
params = utils.get_args_dict(('beta', 'lam', 'T', 'c', 'project_fn', 'dtype'), (beta, 10.0, T, 1e-5, project_fn, dtype))
loss_fn = utils.l2_loss_scaled
gd = descent_structure(data, params, loss_fn)
subgrad_fn = gd.get_torch_subgrad
x_md, error_md, l1_md, xs_md = gd.descent(mirror_descent_KL_prob, subgrad_fn, norm_fn, x_start)
test_errors_md = utils.test_error(xs_md, loss_fn, test_data, params)

print('final estimate:')
print(x_md)
print(torch.sum(x_pgd_2))

# ---------------------------
# Plots + Save
# ---------------------------
plt.clf()
plt.plot(error_pgd, label='Proximal GD (L1)')
plt.title('Training Error')
plt.legend()
plt.savefig('p1/train_error_l1.pdf')

plt.clf()
plt.plot(error_pgd_2, label='Proximal GD (L2)')
plt.plot(error_md, label='Proximal MD (L2)')
plt.title('Training Error')
plt.legend()
plt.savefig('p1/train_error_l2.pdf')

plt.clf()
plt.plot(test_errors_pgd, label='Proximal GD (L1)')
plt.plot(test_errors_pgd_2, label='Proximal GD (L2)')
plt.plot(test_errors_md, label='Proximal MD (L2)')
plt.title("Test Error")
plt.legend()
plt.savefig('p1/test_error.pdf')
