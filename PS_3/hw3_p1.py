import matplotlib.pyplot as plt
from descent_algos import *
import utils

# ==========================================
# HW 3 (Problem 1)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
A_train = np.load("A_train.npy")
b_train = np.load("b_train.npy")
A_test = np.load("A_test.npy")
b_test = np.load("b_test.npy")

# ---------------------------
# Run Descent Algos on LASSO
# ---------------------------

# Set up Descent Structure

T = int(1e2)
alpha, beta = utils.get_alpha_beta(A_train)
data = get_args_dict(('A', 'b'), (A_train, b_train))
parameters = get_args_dict(('beta', 'lam', 'T', 'c'), (beta, 1.0, T, 1e-5))
gd = descent_structure(data, parameters)


# LASSO using Proximal Gradient Descent
subgrad_fn = utils.get_l2_subgrad
x_pgd, error_pgd, l1_pgd, xs_pgd = gd.descent(proximal_gradient_update, subgrad_fn)
test_errors_pgd = test_error(xs_pgd, A_test, b_test)



# LASSO using subgradient method
x_sg, error_sg, l1_sg, xs_sg = gd.descent(subgradient_update, subgrad_fn)
test_errors_sg = test_error(xs_sg, A_test, b_test)
norm_star = l1_sg[-1]


# LASSO using FW methods
gd.parameters['gamma'] = norm_star
x_fw, error_fw, l1_fw, xs_fw = gd.descent(frank_wolfe_update, subgrad_fn)
test_errors_fw = test_error(xs_fw, A_test, b_test)



# ---------------------------
# Plots + Save
# ---------------------------
plt.clf()
plt.plot(error_sg, label='Subgradient')
plt.plot(error_fw, label='Frank-Wolfe')
plt.plot(error_pgd, label='Proximal GD')
plt.title('Training Error')
plt.legend()
plt.savefig('p1/train_error.eps')

plt.clf()
plt.plot(l1_sg, label='Subgradient')
plt.plot(l1_fw, label='Frank-Wolfe')
plt.plot(l1_pgd, label='Proximal GD')
plt.title("$\ell^1$ Norm")
plt.legend()
plt.savefig('p1/l1.eps')

plt.clf()
plt.plot(test_errors_sg, label='Subgradient')
plt.plot(test_errors_fw, label='Frank-Wolfe')
plt.plot(test_errors_pgd, label='Proximal GD')
plt.title("Test Error")
plt.legend()
plt.savefig('p1/test_error.eps')
