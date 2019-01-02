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

# x, y = load_digits(2, True)
x, y = load_breast_cancer(True)

MC = 0.0
DT = 0.0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

    model = MatchingClassifier()
    start_time = time.time()
    model.fit(X_train, y_train)
    print('MC time', time.time() - start_time)

    # z.visualize_tree()

    y_hat = model.predict(X_test)
    MC += metrics.roc_auc_score(y_test, y_hat)
    # print('MC accuracy', )

    model = DecisionTreeClassifier()
    start_time = time.time()
    model.fit(X_train, y_train)
    print('DT time', time.time() - start_time)
    # z.visualize_tree()

    y_hat = model.predict(X_test)
    DT += metrics.roc_auc_score(y_test, y_hat)
    # print('DT accuracy', )
print(MC/100.0)
print(DT/100.0)


xx=0
