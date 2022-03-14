from __future__ import annotations

import math
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        self.mu_ = np.mean(X)
        self.var_ = np.var(X, ddof=not self.biased_)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()
        exp = np.exp(-np.power(X - self.mu_, 2) / 2 * self.var_)
        divisor = math.sqrt(2 * math.pi * self.var_)
        return exp / divisor

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # raise NotImplementedError()
        double_var = 2 * pow(sigma, 2)
        log_of_exp = np.sum(np.power((X - mu), 2)) / -double_var
        lgo_of_factor = -X.size * math.log(math.sqrt(double_var * math.pi))
        return lgo_of_factor + log_of_exp


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: np.ndarray
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: np.ndarray
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        self.mu_ = X.mean(axis=0)
        self.cov_ = np.cov(np.transpose(X), ddof=1)  # ddof=1 used for unbiased estimator
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()
        centered_matrix = X - self.mu_
        tmp = np.dot(centered_matrix, inv(self.cov_))
        exponent = []
        for i in range(X.shape[0]):
            exponent.append(np.dot(tmp[i], centered_matrix[i]))

        exponent = np.array(exponent) / -2
        exp = np.exp(exponent)
        dimensions = X.shape[1]
        divisor = math.sqrt(pow(2 * math.pi, dimensions) * det(self.cov_))
        return exp / divisor

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # raise NotImplementedError()
        dimension = mu.size
        number_of_samples = X.shape[0]
        centered_matrix = X - mu  # from each row of X subtract mu vector
        tmp = np.dot(centered_matrix, inv(cov))
        log_of_exp = 0
        for i in range(tmp.shape[0]):
            log_of_exp += np.dot(tmp[i], centered_matrix[i])
        log_of_exp /= -2

        sign, logdet = slogdet(cov)
        log_of_covariance_det = sign * logdet
        log_of_factor = number_of_samples * dimension * math.log(2 * math.pi) / 2
        log_of_factor += (number_of_samples * log_of_covariance_det / 2)
        return log_of_exp - log_of_factor
