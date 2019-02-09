import matplotlib.pyplot as plt
from descent_algos import *
import utils


# ==========================================
# HW 3 (Problem 2)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
m = 3000
n = 1500
A = np.random.randn(m, n)
x_true = np.random.randn(n, 1)
b = np.matmul(A, x_true).reshape([-1,])

# ---------------------------
# Run Descent Algos on LASSO
# ---------------------------

# Set up Descent Structure
T = int(1e2)
alpha, beta = utils.get_alpha_beta(A)
data = get_args_dict(('A', 'b'), (A, b))
parameters = get_args_dict(('alpha', 'beta', 'lam', 'T', 'c'), (alpha, beta, 1.0, T, 1e-5))
gd = descent_structure(data, parameters)


# LASSO using Proximal Gradient Descent
subgrad_fn = utils.get_l2_subgrad
x_pgd, error_pgd, l1_pgd, xs_pgd = gd.descent(proximal_gradient_update, subgrad_fn)
test_errors_pgd = test_error(xs_pgd, A, b)


# LASSO using subgradient method
x_sg, error_sg, l1_sg, xs_sg = gd.descent(subgradient_update, subgrad_fn)
test_errors_sg = test_error(xs_sg, A, b)


# LASSO using FISTA
ab_tuple = (alpha, beta)
x_F, error_F, l1_F, xs_F = gd.accelerated_descent(FISTA_update, subgrad_fn)


# ---------------------------
# Plots + Save
# ---------------------------
plt.clf()
plt.plot(error_sg, label='Subgradient')
plt.plot(error_pgd, label='Proximal GD')
plt.plot(error_F, label='FISTA')
plt.title('Training Error')
plt.legend()
plt.savefig('p2/train_error.eps')

plt.clf()
plt.plot(l1_sg, label='Subgradient')
plt.plot(l1_pgd, label='Proximal GD')
plt.plot(l1_F, label='FISTA')
plt.title("$\ell^1$ Norm")
plt.legend()
plt.savefig('p2/l1.eps')

