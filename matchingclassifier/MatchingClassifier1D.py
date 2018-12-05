import numpy as np


class MatchingClassifier1D:

    def __init__(self):
        self.num_classes = 0
        self.threshold = 0
        self.coefficient = 0

    def multiclass_optimal_margin(self, data: np.ndarray, classes: np.ndarray):
        '''

        :param data: 1D data
        :param classes: classes (0, 1, ...)
        :return:
        '''
        class_labels = np.unique(classes)
        num_classes = np.unique(classes).size
        self.num_classes = num_classes

        coef = np.zeros((num_classes, num_classes))
        thr = np.zeros((num_classes, num_classes))

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                c1 = class_labels[i]
                c2 = class_labels[j]
                if c1==c2:
                    continue
                condition = (classes == c1) | (classes == c2)
                d = data[condition]
                c = classes[condition]
                c[c==c1] = 0
                c[c==c2] = 1
                m = c.size
                t1 = c.sum()
                t0 = m - t1
                thr[i, j], coef[i, j], _, _ = self.__optimal_discrimination__(d, c, t0, t1)

        return thr, coef

    def fit_1D(self, X, c):
        self.threshold, self.coefficient = self.multiclass_optimal_margin(X, c)

    def predict(self, data):
        '''

        :param data: Is a column vector
        :return:
        '''
        return np.apply_along_axis(self.get_class, axis=1, arr=data)

    def get_class(self, data):
        votes = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                d = self.coefficient[i, j] * data - self.threshold[i, j]
                if d <= 0:
                    votes[j] = votes[j] + 1
                else:
                    votes[i] = votes[i] + 1

        return votes.argmax()


    @staticmethod
    def __optimal_discrimination__(d, c, t0, t1):
        '''

        :param d: 1D data
        :param c: classes (must be 0/1)
        :param t0: number of instances in class 0
        :param t1: number of instances in class 1
        :return: thr: where to cut, coef: which direction, perf: how good it was, marg: what was the final margin
        '''
        d_i = d.argsort()
        ds = d[d_i]
        cs = c[d_i]
        n1ls = np.cumsum(cs)
        n0ls = np.cumsum(-(cs - 1.0))
        l0 = np.divide(n0ls, t0)
        l1 = np.divide(n1ls, t1)
        acc1 = l0 + (1.0 - l1)
        acc2 = 2.0 - acc1  # l1 + (1 - l0)
        ind1 = np.argmax(acc1)
        ind2 = np.argmax(acc2) - 1
        a1 = acc1[ind1]
        a2 = acc2[ind2]
        if a1 > a2:
            thr = -((ds[ind1] + ds[ind1 + 1]) / 2.0)
            coef = -1
            perf = a1
            marg = abs(ds[ind1] - ds[ind1 + 1])
        else:
            thr = ((ds[ind2] + ds[ind2 + 1]) / 2.0)
            coef = 1
            perf = a2
            marg = abs(ds[ind2] - ds[ind2 + 1])
        perf /= 2.0
        perf = 1 - perf
        return thr, coef, perf, marg
