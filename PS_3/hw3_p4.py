import matplotlib.pyplot as plt
from descent_algos import *
import utils
import pandas as pd
import random


# ==========================================
# HW 3 (Problem 4)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
print('============= Loading Data ==============')
nrows = 7000
df = pd.read_csv("Bigdata/X_train.csv", header=-1, nrows=nrows)
df2 = pd.read_csv("Bigdata/y_train.csv", header=-1)
X_train = df.as_matrix()
y_train = df2.as_matrix()
print('=============  Data Loaded ==============')

X_train = X_train[:, :].T
y_train = y_train[0, 0:nrows]
n, N = np.shape(X_train)
num_c = 20

print(y_train.shape)
print(X_train.shape)


# ---------------------------
# Run Descent Algos on Log. Reg.
# ---------------------------

# Set up Descent Structure
T = int(1e3)
mu = 1e-2
alpha, beta = utils.get_alpha_beta(X_train)
data = utils.get_args_dict(('X', 'y'), (X_train, y_train))
parameters = utils.get_args_dict(('alpha', 'beta', 'T', 'mu'), (alpha, beta, T, mu))
gd = descent_structure(data, parameters)
error_fn = utils.get_logist_loss
norm_fn = utils.get_logist_subgrad

fig1 = plt.figure()
ax1 = plt.axes()
fig2 = plt.figure()
ax2 = plt.axes()

print('========== Starting Experiment ==============')

print('------------ Log. Reg. ---------------------')
# Logistic Regression using subgradient method
Beta_start = np.random.randn(n, num_c)
subgrad_fn = utils.get_logist_subgrad
Beta_sg, error_sg, l1_sg, xs_sg =\
    gd.descent(subgradient_update, subgrad_fn, error_fn, norm_fn, Beta_start)
print('preds:')
print(get_logistic_preds(X_train, Beta_sg))

print('-------- Log. Reg. + Acceleration ----------')
# Logistic Regression using accelerated subgradient method
Beta_acc, error_acc, l1_acc, xs_acc =\
    gd.accelerated_descent(accelerated_subgrad_update, subgrad_fn, error_fn, norm_fn, Beta_start)
print('acc preds:')
print(get_logistic_preds(X_train, Beta_acc))

print('true:')
print(y_train)

ax1.plot(error_sg, label='Subgrad_')
ax1.plot(error_acc, label='Accel. Subgrad_')


print('========== END ==============')
# ---------------------------
# Save Plots
# ---------------------------

ax1.legend()
fig1.savefig('p4/train_error.eps')

