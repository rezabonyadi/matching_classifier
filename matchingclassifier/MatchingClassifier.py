import numpy as np


class MatchingClassifier:
    def __init__(self, max_depth=100):
        self.max_depth = max_depth
        self.tree_root = None

    def fit(self, X, classes):
        self.tree_root = self.build_tree(X, classes, 0, self.max_depth)

    def visualize_tree(self):
        self.recursive_visualize_tree(self.tree_root, 0)

    def recursive_visualize_tree(self, root, l):
        if root.class_label is not None:
            t = ''
            for i in range(l):
                t += '   '
            t += ''.join(['Class is ', str(root.class_label)])
            print(t)

        else:
            t = ''
            for i in range(l):
                t += '   '
            t += ''.join(['if x[', str(root.selected_variable), '] <= ', str(root.threshold), ' then '])
            print(t)
            self.recursive_visualize_tree(root.left_node, l+1)
            t = ''
            for i in range(l):
                t += '   '
            t += ''.join('else:')
            print(t)
            self.recursive_visualize_tree(root.right_node, l + 1)

    def predict(self, X):
        labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            root = self.tree_root
            x = X[i, :]
            while root.class_label is None:
                v = x[root.selected_variable]
                if v <= root.threshold:
                    root = root.left_node
                else:
                    root = root.right_node

            labels[i] = root.class_label
        return labels

    def build_tree(self, x, y, current_depth, max_depth):
        n1 = sum(y)
        n0 = y.shape[0] - n1
        perc_1 = n1 / (n0 + n1)
        perc_0 = n0 / (n0 + n1)

        if (current_depth == max_depth) | (y.std() == 0.0):
            if perc_0 > perc_1:
                label = 0
            else:
                label = 1
            node = DecomposerTree(None, None, None, perc_0 + perc_1, perc_0, perc_1, y.shape[0], label)
            return node
        else:
            best_perf, best_dimension, best_thr, best_coef, Lx, Rx, Ly, Ry, _, _ = self.pick_best_dimension(x, y)
            node = DecomposerTree(best_thr, best_coef, best_dimension, best_perf, perc_0, perc_1, y.shape[0], None)
            node.left_node = self.build_tree(Lx, Ly, current_depth + 1, max_depth)
            node.right_node = self.build_tree(Rx, Ry, current_depth + 1, max_depth)
            return node

    def pick_best_dimension(self, X, Y):
        [m, n] = X.shape
        best_perf = -1
        best_dimension = 0
        best_thr = 0.0
        t1 = np.sum(Y)
        t0 = m - t1

        for i in range(n):
            if np.var(X[:, i]) == 0:  # Nothing to distinguish
                continue
            [thr, coef, perf, marg, per_l0, per_l1] = self.__optimal_discrimination__(X[:, i], Y, t0, t1)
            if (perf > best_perf):
                best_perf = perf
                best_thr = thr
                best_coef = coef
                best_marg = marg
                best_per_l0 = per_l0
                best_per_l1 = per_l1
                best_dimension = i

        left_indices = np.where(X[:, best_dimension] <= best_thr)[0]
        right_indices = np.where(X[:, best_dimension] > best_thr)[0]

        LX = X[left_indices, :]
        RX = X[right_indices, :]
        LY = Y[left_indices]
        RY = Y[right_indices]

        return best_perf, best_dimension, best_thr, best_coef, LX, RX, LY, RY, best_per_l0, best_per_l1

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

        n1ls = 0.0
        n0ls = 0.0
        perf = 0.0

        for i in range(cs.shape[0] - 1):
            if cs[i] == 1:
                n1ls += 1
            else:
                n0ls += 1

            if ds[i] == ds[i+1]:
                continue

            l0 = (n0ls / t0)
            l1 = (n1ls / t1)
            acc1 = l0 + (1.0 - l1)  # Accuracy, assuming 0s to the left
            acc2 = l1 + (1.0 - l0)  # Accuracy assuming 1s to the left

            if acc1 > perf:  # 0s to the left
                perf = acc1
                thr = (ds[i] + ds[i+1])/2.0
                coef = 1
                marg = abs(ds[i]-ds[i+1])
                per_l0 = l0
                per_l1 = l1

            if acc2 > perf: # 1s to the left
                perf = acc2
                thr = (ds[i] + ds[i + 1]) / 2.0
                coef = -1
                marg = abs(ds[i] - ds[i + 1])
                per_l0 = l0
                per_l1 = l1

        #
        # n1ls = np.cumsum(cs) # Number of 1s to the left of each value, inclusive
        # n0ls = np.cumsum(-(cs - 1.0)) # Number of 0s to the left of each value, inclusive
        # # Fix for repetitive values
        # for i in range(d_i.shape[0] - 1, 0, -1):
        #     if ds[i] == ds[i - 1]:
        #         n0ls[i-1] = n0ls[i]
        #         n1ls[i-1] = n1ls[i]
        #
        # l0 = np.divide(n0ls, t0)  # Percentage of 0s to the left, corresponding to items in ds
        # l1 = np.divide(n1ls, t1)  # Percentage of 1s to the left, corresponding to items in ds
        # acc1 = l0 + (1.0 - l1)  # Average performance, assuming 0s to the left and 1s to the right
        # acc2 = 2.0 - acc1  # l1 + (1 - l0): Average performance, asusuming 1s to the left and 0s to the right
        # ind1 = np.argmax(acc1) - 1  # Assuming 0s to the left, the index of the best item in ds that achieves the best performance
        # ind2 = np.argmax(acc2) - 1  # Assuming 1s to the left, the index of the best item in ds that achieves the best performance
        # a1 = acc1[ind1]
        # a2 = acc2[ind2]
        # if a1 > a2:  # 0s to the left, inclusinve
        #     thr = ((ds[ind1] + ds[ind1 + 1]) / 2.0)
        #     coef = 1
        #     perf = a1
        #     per_l0 = l0[ind1]
        #     per_l1 = l1[ind1]
        #     marg = abs(ds[ind1] - ds[ind1 + 1])
        #     tind = ind1
        # else:
        #     thr = ((ds[ind2] + ds[ind2 + 1]) / 2.0)
        #     coef = -1
        #     perf = a2
        #     per_l0 = l0[ind2]
        #     per_l1 = l1[ind2]
        #     marg = abs(ds[ind2] - ds[ind2 + 1])
        #     tind = ind2
        perf /= 2.0
        # perf = 1 - perf
        return thr, coef, perf, marg, per_l0, per_l1

class DecomposerTree:
    def __init__(self, t, c, d, p, l0, l1, n, l):
        self.threshold = t # Threshold
        self.coefficient = c # Coefficient: If negative, values <= thr are most likely to be 1s, otherwise 0s
        self.selected_variable = d # Dimension
        self.performance = p # Performance
        self.percentage_of_zero = l0 # Percentage of o
        self.percentage_of_one = l1 # Percentage of 1
        self.number_of_instances = n # Number of instances
        self.right_node = None # Right node
        self.left_node = None # Left node
        self.class_label = l
