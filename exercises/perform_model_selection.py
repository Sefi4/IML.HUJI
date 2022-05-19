from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best
    fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    # x = np.random.uniform(-1.2, 2.001, n_samples)
    x = np.linspace(-1.2, 2.001, n_samples)
    y_ = response(x)
    y = y_ + np.random.normal(scale=noise, size=len(y_))
    trainX, trainY, testX, testY = split_train_test(pd.DataFrame(x), pd.Series(y), 2 / 3)
    trainX, trainY, testX, testY = trainX.to_numpy().flatten(), trainY.to_numpy(), \
                                   testX.to_numpy().flatten(), testY.to_numpy()
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=x, y=y_, mode='markers+lines', name='True Values'),
                    go.Scatter(x=testX, y=testY, mode='markers', name='Train labels'),
                    go.Scatter(x=trainX, y=trainY, mode='markers', name='Test labels')])
    fig.update_layout(title_text="Data set drown from f(x)=(x+3)(x+2)(x+1)(x-1)(x-2)",
                      xaxis_title='X', yaxis_title='Y')
    fig.show()

    # raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores = []
    validation_scores = []
    for k in range(10):
        train_score, validation_score = cross_validate(
            PolynomialFitting(k), trainX, trainY,
            lambda y_true, y_pred, *args: mean_square_error(y_true, y_pred))
        train_scores.append(train_score)
        validation_scores.append(validation_score)

    fig = go.Figure()
    fig.add_traces([go.Scatter(x=np.arange(10), y=train_scores, name='Train Scores'),
                    go.Scatter(x=np.arange(10), y=validation_scores,
                               name='Validation Scores')])
    fig.update_layout(
        title_text="The Average Validation And Train Scores As a Function Of "
                   "Polynomial Degree", xaxis_title='X', yaxis_title='Y')
    fig.show()
    # raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_deg = int(np.argmin(validation_scores))
    pl = PolynomialFitting(best_deg)
    pl.fit(trainX, trainY)
    print("best k is", best_deg, "\nTest error is",
          mean_square_error(testY, pl.predict(testX)))
    # raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    trainX, trainY, testX, testY = split_train_test(X, y, train_proportion=(
                n_samples / y.size))
    trainX, trainY, testX, testY = trainX.to_numpy(), trainY.to_numpy(), \
                                   testX.to_numpy(), testY.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for
    # Ridge and Lasso regressions
    ridge_lambdas = np.linspace(0.001, 0.01, n_evaluations)
    ridge_train_scores = []
    ridge_validation_scores = []

    lasso_lambdas = np.linspace(0.01, 0.2, n_evaluations)
    lasso_train_scores = []
    lasso_validation_scores = []
    for ridge_lambda, lasso_lambda in zip(ridge_lambdas, lasso_lambdas):
        ridge_train_score, ridge_validation_score = cross_validate(
            RidgeRegression(ridge_lambda), X, y,
            lambda y_true, y_pred, *args: mean_square_error(y_true, y_pred))
        ridge_train_scores.append(ridge_train_score)
        ridge_validation_scores.append(ridge_validation_score)

        lasso_train_score, lasso_validation_score = cross_validate(
            Lasso(lasso_lambda), X, y,
            lambda y_true, y_pred, *args: mean_square_error(y_true, y_pred))
        lasso_train_scores.append(lasso_train_score)
        lasso_validation_scores.append(lasso_validation_score)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ridge", "Lasso"))

    fig.add_trace(go.Scatter(x=ridge_lambdas, y=ridge_train_scores, name='Train Scores'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ridge_lambdas, y=ridge_validation_scores,
                             name='Validation Scores'), row=1, col=1)

    fig.add_trace(go.Scatter(x=lasso_lambdas, y=lasso_train_scores, name='Train Scores'),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=lasso_lambdas, y=lasso_validation_scores,
                             name='Validation Scores'), row=1, col=2)

    fig.update_layout(
        title_text="The Average Validation And Train Scores As a Function Of "
                   "regularization parameter", xaxis_title='X', yaxis_title='Y')
    fig.show()
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lambda = ridge_lambdas[np.argmin(ridge_validation_scores)]
    lasso_best_lambda = lasso_lambdas[np.argmin(lasso_validation_scores)]
    ridge_learner = RidgeRegression(ridge_best_lambda)
    lasso_learner = Lasso(lasso_best_lambda)
    linear_reg_learner = LinearRegression()

    ridge_learner.fit(trainX, trainY)
    lasso_learner.fit(trainX, trainY)
    linear_reg_learner.fit(trainX, trainY)

    print("Ridge best lambda is", ridge_best_lambda)
    print("Ridge test error is", ridge_learner.loss(testX, testY))
    print("Lasso best lambda is", lasso_best_lambda)
    print("Lasso test error is", mean_square_error(testY, lasso_learner.predict(testX)))
    print("Linear Regression test error is", linear_reg_learner.loss(testX, testY))


if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(1500, 10)
    select_regularization_parameter()
    # raise NotImplementedError()
