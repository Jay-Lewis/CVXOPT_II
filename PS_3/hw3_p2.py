import matplotlib.pyplot as plt
from descent_algos import *


# ==========================================
# HW 3 (Problem 2)
# ==========================================


# --------------------
# Load Data
# --------------------
m = 1000
n = 400
A = np.random.randn(m, n)
x_true = np.random.randn(n, 1)
b = np.matmul(A, x_true).reshape([-1,])

# ---------------------------
# Run Descent Algos on LASSO
# ---------------------------
T = int(1e2)

# LASSO using Proximal Gradient Descent
alpha, beta = get_alpha_beta(A)
print(alpha, beta)
x_pgd, error_pgd, l1_pgd, xs_pgd =\
    descent(proximal_gradient_update, A, b, reg=1, T=T, c=beta)

# LASSO using subgradient
x_sg, error_sg, l1_sg, xs_sg = \
    descent(subgradient_update, A, b, reg=1, T=T, c=1e-5)

# LASSO using FISTA
x_F, error_F, l1_F, xs_F = accelerated_descent(FISTA_update, A, b, reg=1, T=T, c=(alpha, beta))

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
plt.savefig('p2/train_error.png')

plt.clf()
plt.plot(l1_sg, label='Subgradient')
plt.plot(l1_pgd, label='Proximal GD')
plt.plot(l1_F, label='FISTA')
plt.title("$\ell^1$ Norm")
plt.legend()
plt.savefig('p2/l1.eps')
plt.savefig('p2/l1.png')

