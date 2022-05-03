from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, f1_score
import plotly.graph_objects as go


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.lg = LogisticRegression(penalty='none')
        # self.nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
        #                         hidden_layer_sizes=(5, 2), random_state=1)
        # self.lda = LinearDiscriminantAnalysis()
        # self.knn = KNeighborsClassifier(5)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        self.lg.fit(X, y)
        # self.nn.fit(X, y)
        # self.lda.fit(X, y)
        # self.knn.fit(X, y)

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
        # return self.lda.predict(X)
        # return self.nn.predict(X)
        return self.lg.predict(X)
        # return self.knn.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        fpr, tpr, thresholds = roc_curve(y, self.predict(X))
        print(f1_score(y, self.predict(X)))
        # return f1_score(y, self.predict(X))

        go.Figure(
            data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="black", dash='dash'),
                             name="Random Class Assignment"),
                  go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                             name="", showlegend=False, marker_size=5,
                             hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
            layout=go.Layout(
                title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
        return auc(fpr, tpr)