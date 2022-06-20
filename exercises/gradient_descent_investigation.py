import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from sklearn.metrics import roc_curve, auc
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_best_eta = -1
    l2_best_eta = -1
    min_loss_l1 = float('inf')
    min_loss_l2 = float('inf')
    fig1 = go.Figure(layout=dict(
        title=f"Convergence Rate of L1 As A Function Of GD Iterations over different etas"))
    fig2 = go.Figure(layout=dict(
        title=f"Convergence Rate of L2 As A Function Of GD Iterations over different etas"))
    for eta in etas:
        callback1, l1_values, l1_weights = get_gd_state_recorder_callback()
        callback2, l2_values, l2_weights = get_gd_state_recorder_callback()
        GradientDescent(FixedLR(eta), callback=callback1).fit(L1(init))
        GradientDescent(FixedLR(eta), callback=callback2).fit(L2(init))

        l1_values, l1_weights = np.array(l1_values), np.array(l1_weights)
        l2_values, l2_weights = np.array(l2_values), np.array(l2_weights)

        if eta == .01:
            plot_descent_path(L1, l1_weights, f"L1 Descent Path with learning rate {eta}").show()
            plot_descent_path(L2, l2_weights, f"L2 Descent Path with learning rate {eta}").show()

        fig1.add_trace(go.Scatter(x=np.arange(1000), y=l1_values, mode='markers', name=f'eta {eta}'))
        fig2.add_trace(go.Scatter(x=np.arange(1000), y=l2_values, mode='markers', name=f'eta {eta}'))

        if l1_values[-1] < min_loss_l1:
            l1_best_eta = eta
            min_loss_l1 = l1_values[-1]

        if l2_values[-1] < min_loss_l2:
            l2_best_eta = eta
            min_loss_l2 = l2_values[-1]

    fig1.show()
    fig2.show()
    print(f"lowest loss of L1 norm with eta {l1_best_eta}", min_loss_l1)
    print(f"lowest loss of L2 norm with eta {l2_best_eta}", min_loss_l2)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure(layout=dict(
        title=f"Convergence Rate of L1 As A Function Of GD Iterations over different gammas"))
    min_loss = float('inf')
    best_gamma = -1
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        GradientDescent(ExponentialLR(eta, gamma), callback=callback).fit(L1(init))

        values, weights = np.array(values), np.array(weights)
        fig.add_trace(go.Scatter(x=np.arange(1000), y=values, mode='markers+lines', name=f'gamma {gamma}'))

        if values[-1] < min_loss:
            best_gamma = gamma
            min_loss = values[-1]

        if gamma == .95:
            plot_descent_path(L1, weights,
                              f"L1 Descent Path with learning rate {eta} and gamma {gamma}").show()

    fig.show()
    print(f"lowest loss of L1 norm with eta {eta} and gamma {best_gamma}", min_loss)


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset

    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    solver = GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4))
    lg = LogisticRegression(penalty='l1', solver=solver)
    lg.fit(X_train, y_train)
    proba = lg.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, proba)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    best_alpha = thresholds[np.argmax(tpr - fpr)]
    print("Best alpha", best_alpha)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
    for i in range(2):
        train_scores = []
        validation_scores = []
        for lam in lambdas:
            train_score, validation_score = cross_validate(
                LogisticRegression(penalty=f'l{i + 1}', solver=solver, lam=lam),
                X_train,
                y_train,
                misclassification_error)
            train_scores.append(train_score)
            validation_scores.append(validation_score)

        best_lam = lambdas[int(np.argmin(validation_scores))]
        lg = LogisticRegression(penalty=f'l{i + 1}', solver=solver, lam=best_lam)
        lg.fit(X_train, y_train)
        print(f"Best lambda for L{i + 1} norm", best_lam)
        print(f"Loss of L{i + 1} norm with best lbamgda", lg.loss(X_test, y_test))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
