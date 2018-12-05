from matchingclassifier.MatchingClassifier1D import MatchingClassifier1D
import numpy as np

class MatchingClassifier:

    def __init__(self):
        pass


    def fit(self, X, classes):
        prediction_votes = np.zeros(classes.shape[0])

        for z in range(10):
            a, b = self.pick_one_dimension(X, classes, [], prediction_votes)

    def pick_one_dimension(self, X, classes, currently_picked: list, prediction_votes):
        mc1d = MatchingClassifier1D()

        temp_prediction_votes = np.zeros(classes.shape[0])
        best_perf = 0
        for i in range(X.shape[0]):
            d = X[i, :]
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


