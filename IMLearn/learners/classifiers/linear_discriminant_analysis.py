from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.mu_ = []
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for k in range(self.classes_.size):
            indices = np.where(y == self.classes_[k])
            self.mu_.append(np.mean(X[indices], axis=0))
            self.cov_ += np.cov(X[indices], rowvar=False, ddof=0) * len(indices)

        self.cov_ /= (X.shape[0] - len(self.classes_))
        print('cov =', self.cov_)
        print()
        self._cov_inv = inv(self.cov_)
        self.mu_ = np.array(self.mu_)
        print('mu =', self.mu_)
        self.pi_ = counts / y.size
        # raise NotImplementedError()

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        tmp = []
        for k in range(self.classes_.size):
            mu_t = np.transpose(self.mu_[k])
            a = self._cov_inv @ mu_t
            b = np.log(self.pi_[k]) - (self.mu_[k] @ self._cov_inv @ mu_t) / 2
            tmp.append(a.T @ X.T + b)

        tmp = np.array(tmp)
        indices = np.argmax(tmp, axis=0)
        return np.array(list(map(lambda i: self.classes_[i], indices)))
        # raise NotImplementedError()

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        d = X[np.newaxis, :, :]

        # mahalanobis = np.sum(d.dot(inv(cov)) * d)
        raise NotImplementedError()

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
        # raise NotImplementedError()
