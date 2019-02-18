
import matplotlib.pyplot as plt
from descent_algos import *
import pandas as pd
from sklearn.linear_model import LogisticRegression
import sklearn_p4_utils as skutils


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
nrows = 15000
df = pd.read_csv("Bigdata/X_train.csv", header=-1, nrows=nrows)
df2 = pd.read_csv("Bigdata/y_train.csv", header=-1)
X_train = df.as_matrix()
y_train = df2.as_matrix()
y_train = y_train[0, 0:nrows]
num_c = 20
n, N = np.shape(X_train)
print('=============  Data Loaded ==============')


# ---------------------------
# Run Descent Algos on Log. Reg.
# ---------------------------

fig1 = plt.figure()
ax1 = plt.axes()
fig2 = plt.figure()
ax2 = plt.axes()

print('========== Starting Experiment 1 ==============')

print('------------ Log. Reg. ---------------------')
# Logistic Regression using subgradient method
Beta_start = np.random.randn(n, num_c)
T = 10
num_pts = 5

# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', verbose=0)
# filename = 'p4/mu_sweep_subgrad.pdf'
# skutils.plot_loss_vs_reg("C", -8, 3, num_pts, clf, X_train, y_train, filename)

print('-------- Log. Reg. + Acceleration ----------')
# Logistic Regression using accelerated subgradient method
clf2 = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial', verbose=0, max_iter=25)
filename = 'p4/mu_sweep_acc_subgrad.pdf'
skutils.plot_loss_vs_reg("C", -8, 3, num_pts, clf2, X_train, y_train, filename)

print('========== END ==============')



# print('========== Starting Experiment 2 ==============')
#
# print('------------ Log. Reg. ---------------------')
#
# clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
# print(np.shape(clf.coef_))
#
# print('-------- Log. Reg. + Acceleration ----------')
# # Logistic Regression using accelerated subgradient method
# clf2 = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(X_train, y_train)
#
# ax1.plot(error_sg, label='Subgrad_')
# ax1.plot(error_acc, label='Accel. Subgrad_')
#
#
# print('========== END ==============')




# print('=======================TEST===================================')
#
# print('============= Loading Data ==============')
# nrows = 100
# df = pd.read_csv("Bigdata/X_test.csv", header=-1, nrows=nrows)
# df2 = pd.read_csv("Bigdata/y_test.csv", header=-1)
# X_test = df.as_matrix()
# y_test = df2.as_matrix()
# y_test = y_test[0, 0:nrows]
# num_c = 20
# n, N = np.shape(X_test)
# print('=============  Data Loaded ==============')
#
# preds = clf.predict(X_test)
# print('Estimates:')
# print(preds)
# print('True Labels:')
# print(y_test)
#
# count = [1 for pred, y in zip(preds, y_test) if pred == y]
# print('accuracy', np.average(count))


# ---------------------------
# Save Plots
# ---------------------------

# ax1.legend()
# fig1.savefig('p4/train_error.eps')

