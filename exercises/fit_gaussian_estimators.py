from typing import Tuple
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def plot_pdf(uni_gauss: UnivariateGaussian, samples: np.ndarray,
             show: bool = True) -> None:
    """
        Plot the PDF of uni-variate Gaussian random variable.
        X-axis describes the samples.
        Y-axis describes the probability density.

        uni_gauss:  UnivariateGaussian
            An instance of UnivariateGaussian.

        samples: samples: np.ndarray
            The samples to calculate the pdf

        show: bool
            Whether to show the graph or not.
    """
    fig = go.Figure()
    x_axis = samples
    y_axis = uni_gauss.pdf(samples)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_axis,
        name="Name of Trace 1",  # this sets its legend entry
        mode="markers"
    ))

    fig.update_layout(
        title="Probability density function of fitted univariate Gaussian",
        xaxis_title="Samples",
        yaxis_title="PDF",
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=22,
            color="RebeccaPurple"
        )
    )
    if show:
        fig.show()


def empirical_consistent_univarient_gaussian_estimator(
        samples: np.ndarray, show: bool = True) -> None:
    """
    Plot the absolute distance between the estimated expectation to the true
    expectation of a uni-variate Gaussian random variable over increasing
    samples.
    X-axis describes the samples.
    Y-axis describes the absolute distance between the estimated expectation to
    the true expectation.

    samples: np.ndarray.
        The pool of samples to draw from increasingly number of samples.

    show: bool
        Whether to show the graph or not.
    """
    uni_gauss = UnivariateGaussian()
    fig = go.Figure()
    x_axis = []
    y_axis = []
    mu, sigma = 10, 1

    for i in range(10, 1001, 10):
        x_axis.append(i)
        estimated_mu = uni_gauss.fit(samples[:i]).mu_
        y_axis.append(abs(estimated_mu - mu))

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_axis,
        name="Name of Trace 1"  # this sets its legend entry
    ))

    fig.update_layout(
        title="The absolute distance between estimated and true value of "
              "expectation",
        xaxis_title="Sample Size",
        yaxis_title="The absolute error of mean",
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    if show:
        fig.show()


def generate_log_likelihood_heat_map(
        samples: np.ndarray, cov_matrix: np.ndarray, show: bool = True) \
        -> Tuple[float, float]:
    """
    Plot heat map of the log-likelihood of a multivariate normal
    distribution mean.
    Mu is of the form [f1, 0, f3, 0] where f1 and f3 have 200 evenly spaced
    values in the interval [-10, 10].
    In addition, print the maximus log-likelihood.
    The covariance matrix is given.
    X-axis describes f1 value.
    Y-axis describes f2 value.

    samples: np.ndarray.
        The samples to calculate over the log-likelihood.

    cov_matrix: np.ndarray.
        The covariance matrix to calculate over the log-likelihood.

    show: bool.
        Whether to show the graph or not.

    return: Tuple[float, float].
        The values of f1 and f3 that maximize the log-likelihood.
    """
    x_axis = y_axis = np.linspace(-10, 10, 200)
    mu = np.zeros(4)
    z_axis = []
    for i in range(len(x_axis)):
        mu[0] = x_axis[i]
        row = []
        for j in range(len(y_axis)):
            mu[2] = x_axis[j]
            row.append(
                MultivariateGaussian.log_likelihood(mu, cov_matrix, samples))
        z_axis.append(row)

    fig = go.Figure(go.Heatmap(x=x_axis, y=y_axis, z=z_axis))

    fig.update_layout(
        title="log-likelihood heat map of multivariate gaussian",
        xaxis_title="first coordinate of mean vector",
        yaxis_title="third coordinate of mean vector",
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    if show:
        fig.show()
    max_ind = np.argmax(z_axis)
    x_axis_ind = max_ind // x_axis.size
    y_axis_ind = max_ind % y_axis.size
    print("Max of log-likelihood = ", z_axis[x_axis_ind][y_axis_ind])
    return x_axis[x_axis_ind], y_axis[y_axis_ind]


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # raise NotImplementedError()
    uni_gauss = UnivariateGaussian()
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)
    uni_gauss.fit(samples)
    print((uni_gauss.mu_, uni_gauss.var_))

    # Question 2 - Empirically showing sample mean is consistent
    empirical_consistent_univarient_gaussian_estimator(samples)
    # raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model
    plot_pdf(uni_gauss, samples)
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov_matrix = np.array([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mean, cov_matrix, 1000)
    multi_gaussian = MultivariateGaussian()
    multi_gaussian.fit(samples)
    print("Mean vector:", multi_gaussian.mu_)
    print()
    print("Covariance matrix:", multi_gaussian.cov_)
    print()
    # raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    x, y = generate_log_likelihood_heat_map(samples, cov_matrix)

    # Question 6 - Maximum likelihood
    print()
    print((x, y))
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

