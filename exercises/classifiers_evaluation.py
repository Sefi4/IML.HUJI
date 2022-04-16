import pandas as pd
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IMLearn.metrics import accuracy

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    y = data[:, 2]
    X = data[:, :2]
    return X, y
    # raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, Y = load_dataset(f'C:\Projects\IML\IML.HUJI\datasets\{f}')
        # raise NotImplementedError()

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        p = Perceptron(callback=lambda p, _, __: losses.append(p.loss(X, Y)))
        p.fit(X, Y)
        # raise NotImplementedError()

        # Plot figure
        fig = go.Figure()
        x_axis = list(range(1, len(losses)))
        fig.add_scatter(x=x_axis, y=losses)
        fig.update_layout(title=f'Preceptron losss over {f} data as a function '
                                f'of training iterations', xaxis_title='iterations',
                          yaxis_title='loss')
        fig.show()
        # raise NotImplementedError()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f'C:\Projects\IML\IML.HUJI\datasets\{f}')
        # raise NotImplementedError()

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        lda_pred = lda.predict(X)

        g = GaussianNaiveBayes()
        g.fit(X, y)
        g_pred = g.predict(X)

        # raise NotImplementedError()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'LDA, accuracy: '
                                            f'{accuracy(y, lda_pred)}',
                                            f'Gaussian Naive Bayes, accuracy'
                                            f' {accuracy(y, g_pred)}'))
        fig.add_trace(
            go.Scatter(
                x=X[:, 0], y=X[:, 1], mode="markers",
                showlegend=False,
                marker=dict(color=lda_pred, symbol=y,
                            line=dict(color="black", width=1))),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(
                x=X[:, 0], y=X[:, 1], mode="markers",
                showlegend=False,
                marker=dict(color=g_pred, symbol=y,
                            line=dict(color="black", width=1))),
            row=1, col=2)

        mu = []
        for k in range(len(lda.classes_)):
            X_k = X[g_pred == lda.classes_[k]]
            mu.append(np.mean(X_k, axis=0))
        mu = np.array(mu)

        fig.add_trace(
            go.Scatter(x=mu[:, 0], y=mu[:, 1], mode="markers", marker_size=15,
                       showlegend=False,
                       marker=dict(color='black', symbol=104)),
            row=1, col=1)

        fig.update_layout(title_text=f)
        fig.show()
        # raise NotImplementedError()

if __name__ == '__main__':
    np.random.seed(0)
# run_perceptron()
compare_gaussian_classifiers()
