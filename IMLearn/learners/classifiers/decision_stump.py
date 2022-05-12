from __future__ import annotations
from typing import Tuple, NoReturn
import pandas as pd
from IMLearn.metrics import misclassification_error
from IMLearn.base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # raise NotImplementedError()
        sorted_X_indices = np.argsort(X, axis=0)
        X = np.take_along_axis(X, sorted_X_indices, axis=0)
        min_threshold_err = float('inf')
        for j, sign in product(np.arange(X.shape[1]), [-1, 1]):
            indices = sorted_X_indices[:, j]
            thr, thr_error = self._find_threshold(X[:, j], y[indices], sign)
            if thr_error < min_threshold_err:
                self.j_ = j
                self.threshold_ = thr
                self.sign_ = sign
                min_threshold_err = thr_error
            if thr_error == 0:
                break

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # raise NotImplementedError()
        mask = X[:, self.j_] < self.threshold_
        y_pred = np.full(X.shape[0], self.sign_)
        y_pred[mask] = -self.sign_
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> \
            Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as
        `-sign` whereas values which equal to or above the threshold are predicted as
        `sign`
        """
        # raise NotImplementedError()
        y_pred = np.full(values.size, sign)
        weights = np.abs(labels)
        y = np.sign(labels)
        min_thr_err = float('inf')
        thr = None
        for i in range(y_pred.size + 1):
            if (0 < i < y_pred.size) and values[i] == values[i-1]:
                y_pred[i] = -sign
                continue
            mask = y_pred != y
            thr_err = np.sum(weights[mask] / labels.size)
            if thr_err < min_thr_err:
                min_thr_err = thr_err
                thr = values[i] if i < y_pred.size else np.inf
            if i < y_pred.size:
                y_pred[i] = -sign
        return thr, min_thr_err

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
        # raise NotImplementedError()
        return misclassification_error(y, self.predict(X))

