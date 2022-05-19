from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # raise NotImplementedError()
    validation_scores = []
    train_scores = []
    groups = np.remainder(np.arange(X.shape[0]), cv)
    for k in range(cv):
        mask = k != groups
        trainX, trainY = X[mask], y[mask]
        validateX, validateY = X[~mask], y[~mask]
        estimator.fit(trainX, trainY)
        validation_scores.append(scoring(validateY, estimator.predict(validateX), None))
        train_scores.append(scoring(trainY, estimator.predict(trainX), None))
    validation_scores, train_scores = np.array(validation_scores), np.array(train_scores)
    return float(np.mean(train_scores)), float(np.mean(validation_scores))


