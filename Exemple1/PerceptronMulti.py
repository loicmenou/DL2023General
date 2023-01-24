import numpy as np
from Perceptron import Perceptron
import itertools

class PerceptronMulti(object):
    """Perceptron classifier for multiple classes.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    perceptrons : list
        Perceptrons for each class fitted using one vs all approach.
    labels : list
        Given labels, kept to be returned when asked to predict.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.perceptrons = []
        self.labels = []
        self.label_pairs = []

    def fit_one_vs_one(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        # Count number of class
        self.labels = np.unique(y)
        self.label_pairs = list(itertools.combinations(self.labels, 2))

        for labelPair in self.label_pairs:
            label = labelPair[0]
            sub_y = []
            sub_X = []
            for xi, yi in zip(X, y):
                if yi in labelPair:
                    sub_y.append(1 if yi == label else -1)
                    sub_X.append(xi)

            ppn = Perceptron(eta=self.eta, n_iter=self.n_iter)
            ppn.fit(np.array(sub_X), np.array(sub_y))
            self.perceptrons.append(ppn)

        return self

    def predict_one_vs_one(self, X):
        """Return most positive class label"""
        scores = dict.fromkeys(self.labels, 0)
        for i in range(0, len(self.label_pairs)) :
            score = self.perceptrons[i].net_input(X)

            scores[self.label_pairs[i][0]] += score # if positive, more vote for first class
            scores[self.label_pairs[i][1]] -= score # if negative, more vote for second class

        return max(scores, key = scores.get) # Return label with max score

    def fit_one_vs_all(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        # Count number of class
        self.labels = np.unique(y)

        for label in self.labels :
            binary_y = np.where(y == label, 1, -1)
            ppn = Perceptron(eta=self.eta, n_iter=self.n_iter)
            ppn.fit(X, binary_y)
            self.perceptrons.append(ppn)

        return self

    def predict_one_vs_all(self, X):
        """Return all positive class label"""
        scores = dict.fromkeys(self.labels, 0)
        for i in range(0, len(self.labels)) :
            scores[self.labels[i]] = self.perceptrons[i].net_input(X)

        return max(scores, key = scores.get) # Return label with max score

    def score_one_vs_one(self, X_test, y_test):
        assert len(X_test) == len(y_test)

        correct = 0
        for i in range(0, len(y_test)):
            true_label = y_test[i]

            assert true_label in self.labels

            prediction = self.predict_one_vs_one(X_test[i])
            if true_label == prediction:
                correct += 1

        return correct / len(X_test)

    def score_one_vs_all(self, X_test, y_test):
        assert len(X_test) == len(y_test)

        correct = 0
        for i in range(0, len(y_test)):
            true_label = y_test[i]

            assert true_label in self.labels

            prediction = self.predict_one_vs_all(X_test[i])
            if true_label == prediction:
                correct += 1
        return correct / len(X_test)