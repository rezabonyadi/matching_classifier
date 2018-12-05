from matchingclassifier.MatchingClassifier1D import MatchingClassifier1D
import numpy as np
# from sklearn import metrics

class MatchingClassifier:

    def __init__(self):
        pass


    def fit(self, X, classes):
        number_of_variables = X.shape[1]
        prediction_votes = np.zeros(classes.shape[0])
        mc1d = MatchingClassifier1D()
        performances = np.zeros(number_of_variables)
        for i in range(number_of_variables):
            d = X[:, i]
            mc1d.fit_1D(d, classes)
            c_hat = mc1d.predict(d)
            # performances[i] = metrics.accuracy_score(c_hat, classes)

            # diff = c_hat - classes

            # a, b = self.pick_one_dimension(X, classes, [], prediction_votes)
        zz = 0
    def pick_one_dimension(self, X, classes, currently_picked: list, prediction_votes):
        mc1d = MatchingClassifier1D()

        temp_prediction_votes = np.zeros(classes.shape[0])
        best_perf = 0
        for i in range(X.shape[1]):
            d = X[:, i]
            mc1d.fit_1D(d, classes)
            c_hat = mc1d.predict(d)
            diff = c_hat - classes
            correct = diff == 0
            incorrect = diff != 0
            temp_prediction_votes[correct] = prediction_votes[correct] + 1
            temp_prediction_votes[incorrect] = prediction_votes[incorrect] - 1

            perf = sum(temp_prediction_votes > 0) / classes.shape[0]
            if (best_perf > perf):
                best_perf = perf
                best_index = i

        return best_index, best_perf


