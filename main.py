import numpy as np
from matchingclassifier.MatchingClassifier import MatchingClassifier
from sklearn import metrics

# from sklearn.datasets import fetch_openml

# from keras.datasets import mnist
# from keras import backend as K
# import keras

#
# X = np.random.rand(100)
# X = np.concatenate((X, np.random.rand(100)+4.1), axis=0)
# X = np.concatenate((X, np.random.rand(100)+5), axis=0)
#
# c = np.zeros(100)
# c = np.concatenate((c, np.ones(100)), axis=0)
# c = np.concatenate((c, np.ones(100)*2), axis=0)
#
# X = np.reshape(X, (-1, 1))
# c = np.reshape(c, (-1, 1))
#
# mc = MatchingClassifier1D()
# mc.fit_1D(X, c)
#
# y = mc.predict(X)
# print(metrics.accuracy_score(y, c))

from sklearn.datasets import load_digits, load_breast_cancer
# x, y = load_digits(2, True)
x, y = load_breast_cancer(True)

z = MatchingClassifier()
z.fit(x, y)
z.visualize_tree()

cs = z.predict(x)
print(metrics.roc_auc_score(y, cs))
xx=0
