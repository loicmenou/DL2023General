import numpy as np
import Perceptron as p


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

    def fit(self, X, y):
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
            binary_y = np.where(y == label, -1, 1)
            self.perceptrons.append(p.Perceptron(eta=0.1, n_iter=10))
            self.perceptrons[-1].fit(X, binary_y)

        return self

    def predict(self, X):
        """Return all positive class label after unit step"""
        res = []
        for i in range(0, len(self.labels)) :
            if self.perceptrons[i].predict(X) == 1 :
                res.append(self.labels[i])

        return res
