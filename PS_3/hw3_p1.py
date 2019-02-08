import matplotlib.pyplot as plt
from descent_algos import *
from utils import get_alpha_beta

# ==========================================
# HW 3 (Problem 1)
# ==========================================


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
# LASSO using Proximal Gradient Descent
T = int(1e2)
alpha, beta = get_alpha_beta(A_train)
x_pgd, error_pgd, l1_pgd, xs_pgd =\
    descent(proximal_gradient_update, A_train, b_train, reg=1, T=T, c=beta)
test_errors_pgd = test_error(xs_pgd, A_test, b_test)

# LASSO using subgradient and FW methods
x_sg, error_sg, l1_sg, xs_sg = \
    descent(subgradient_update, A_train, b_train, reg=1, T=T, c=1e-5)
test_errors_sg = test_error(xs_sg, A_test, b_test)
norm_star = l1_sg[-1]
x_fw, error_fw, l1_fw, xs_fw = \
    descent(frank_wolfe_update, A_train, b_train, reg=norm_star, T=T)
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
plt.savefig('p1/train_error.png')

plt.clf()
plt.plot(l1_sg, label='Subgradient')
plt.plot(l1_fw, label='Frank-Wolfe')
plt.plot(l1_pgd, label='Proximal GD')
plt.title("$\ell^1$ Norm")
plt.legend()
plt.savefig('p1/l1.eps')
plt.savefig('p1/l1.png')

plt.clf()
plt.plot(test_errors_sg, label='Subgradient')
plt.plot(test_errors_fw, label='Frank-Wolfe')
plt.plot(test_errors_pgd, label='Proximal GD')
plt.title("Test Error")
plt.legend()
plt.savefig('p1/test_error.eps')
plt.savefig('p1/test_error.png')
