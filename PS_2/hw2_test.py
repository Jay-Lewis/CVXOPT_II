# ==========================
# CrossValidation Test for Lasso
# ==========================

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

X = np.load("A.npy")
y = np.load("b.npy")

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)



# tuned_parameters = [{'alpha': alphas}]
# n_folds = 5
#
# clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
# clf.fit(X, y)
# scores = clf.cv_results_['mean_test_score']
# scores_std = clf.cv_results_['std_test_score']
# plt.figure().set_size_inches(8, 6)
# plt.semilogx(alphas, scores)
#
# # plot error lines showing +/- std. errors of the scores
# std_error = scores_std / np.sqrt(n_folds)
#
# plt.semilogx(alphas, scores + std_error, 'b--')
# plt.semilogx(alphas, scores - std_error, 'b--')
#
# # alpha=0.2 controls the translucency of the fill color
# plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
#
# plt.ylabel('CV score +/- std error')
# plt.xlabel('alpha')
# plt.axhline(np.max(scores), linestyle='--', color='.5')
# plt.xlim([alphas[0], alphas[-1]])
#
# plt.show()