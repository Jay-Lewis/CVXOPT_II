import matplotlib.pyplot as plt
from descent_algos import *
import utils


# ==========================================
# HW 3 (Problem 3)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
m = 1000
n = 500
A = np.random.randn(m, n)
x_true = np.random.randn(n,)
b = np.matmul(A, x_true).reshape([-1,])

# ---------------------------
# Run Descent Algos on Log. Reg.
# ---------------------------

# Set up Descent Structure
T = int(1e4)
alpha, beta = utils.get_alpha_beta(A)
data = utils.get_args_dict(('A', 'b'), (A, b))
parameters = utils.get_args_dict(('alpha', 'beta', 'T'), (alpha, beta, T))
gd = descent_structure(data, parameters)
error_fn = utils.get_prob3_loss
norm_fn = utils.get_l1_loss

fig1 = plt.figure()
ax1 = plt.axes()
fig2 = plt.figure()
ax2 = plt.axes()

print('========== Starting Experiment ==============')
for i in range(0, 3):
    print('iteration: ', str(i))
    print('------------ Log. Reg. ---------------------')
    # Logistic Regression using subgradient method
    x_start = np.random.randn(n,)
    subgrad_fn = utils.get_prob3_subgrad
    x_sg, error_sg, l1_sg, xs_sg =\
        gd.descent(subgradient_update, subgrad_fn, error_fn, norm_fn, x_start)

    print('-------- Log. Reg. + Acceleration ----------')
    # Logistic Regression using accelerated subgradient method
    x_F, error_F, l1_F, xs_F =\
        gd.accelerated_descent(accelerated_subgrad_update, subgrad_fn, error_fn, norm_fn, x_start)

    ax1.plot(error_sg, label='Subgrad_'+str(i))
    ax1.plot(error_F, label='Accel. Subgrad_'+str(i))
    ax2.plot(l1_sg, label='Subgradient_'+str(i))
    ax2.plot(l1_F, label='Accel. Subgrad_'+str(i))

print('========== END ==============')
# ---------------------------
# Save Plots
# ---------------------------

ax1.legend()
fig1.savefig('p3/train_error.eps')

