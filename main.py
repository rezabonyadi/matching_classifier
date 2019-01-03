import numpy as np
from matchingclassifier.MatchingClassifier import MatchingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import time
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split

# from sklearn.datasets import fetch_openml

# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x, y = load_digits(10, True)
c1 = 4
c2 = 7
x = x[((y==c1) | (y==c2)),:]
y = y[((y==c1) | (y==c2))]
y[y==c1] = 0
y[y==c2] = 1

# x, y = load_breast_cancer(True)

MC = 0.0
DT = 0.0
nf_m = 0.0
nf_d = 0.0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, stratify=y)

    model = MatchingClassifier(tol=0.05, max_depth=30, extra_sparse=True)
    start_time = time.time()
    model.fit(X_train, y_train)
    print('MC time', time.time() - start_time)
    nf_m += len(model.important_dimensions)
    print('MT #nodes', len(model.important_dimensions))
    # model.visualize_tree()

    y_hat = model.predict(X_test)
    m = metrics.roc_auc_score(y_test, y_hat)
    MC += m
    print('MC accuracy', m)

    model = DecisionTreeClassifier()
    start_time = time.time()
    model.fit(X_train, y_train)
    print('DT time', time.time() - start_time)
    nf_d += model.feature_importances_[model.feature_importances_ > 0].shape[0]
    print('DT #nodes',model.feature_importances_[model.feature_importances_>0].shape[0])
    # z.visualize_tree()

    y_hat = model.predict(X_test)
    d = metrics.roc_auc_score(y_test, y_hat)
    DT += d
    print('DT accuracy', d)

print(MC/100.0)
print(DT/100.0)
print(nf_m/100.0)
print(nf_d/100.0)


xx=0
