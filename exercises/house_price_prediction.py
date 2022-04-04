import os
import colorama
from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils.utils import split_train_test
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # raise NotImplementedError()
    houses_df = pd.read_csv(filename, sep=',')

    # convert the date to the day in the year in {1,2,...,365}
    houses_df['date'] = pd.to_datetime(houses_df.date, format='%Y%m%dT%H%M%S',
                                       errors='coerce').dt.dayofyear
    # drop id feature.
    houses_df = houses_df.drop('id', axis=1)

    # Handle categorical data
    dummies = pd.get_dummies(houses_df['zipcode'], drop_first=True)
    houses_df = pd.concat([houses_df, dummies], axis=1)
    houses_df = houses_df.drop('zipcode', axis=1)

    # remove rows with unreasonable values
    for feature in houses_df:
        if feature in ("price", 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                       'yr_built', 'sqft_living15', 'sqft_lot15'):
            houses_df = houses_df[houses_df[feature] > 0]

    houses_df = houses_df.dropna()
    prices = houses_df.get('price')
    houses_df = houses_df.drop('price', axis=1)
    return houses_df, prices


def get_features_to_response_correlation(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Calculate the correlation of each feature with the target variable y.
    Parameters
    ----------
    X: pd.DataFrame
        The design matrix.

    y: pd.Series
        The target variable.

    Returns
    -------
     pandas.Series of the correlation of each feature with the target variable y.
    """
    y_std = y.std()
    return X.apply(lambda col: col.cov(y) / (col.std() * y_std))


def pick_k_most_correlated_features(X: pd.DataFrame, y: pd.Series,
                                    k: int) -> pd.DataFrame:
    """
    Find the k most correlated features with the target variable.
    ----------
    X: pd.DataFrame
        The design matrix.

    y: pd.Series
        The target variable.

    k: int
        The number of features to take.

    Returns
    -------
     pd.DataFrame with the k most correlated features.
    """
    feature_response_corr = get_features_to_response_correlation(X, y)
    if k >= feature_response_corr.shape[0]:
        return X
    feature_response_corr.sort_values(inplace=True, ascending=False)
    return X[feature_response_corr.index.values[:k]]


def plot_mse(train_X: pd.DataFrame, train_Y: pd.Series, test_X: pd.DataFrame,
             test_Y: pd.Series):
    """
    Plot the MSE over increasing number of samples.
    ----------
    train_X: pd.DataFrame
        Training data st.

    train_Y: pd.DataFrame
        Training target variable.

    test_X: pd.DataFrame
        Test data frame.

    test_Y: pd.DataFrame
        Test target variable.
    """
    lr = LinearRegression(True)
    fig = go.Figure()
    x_axis = np.linspace(10, 100, 91, dtype=int)
    mean_loss, std_loss = [], []
    for p in range(10, 101):
        loss = []
        for i in range(10):
            X = train_X.sample(frac=p / 100)
            Y = train_Y.loc[X.index.values]
            lr.fit(X.to_numpy(), Y)
            tmp = lr.loss(test_X.to_numpy(), test_Y.to_numpy())
            loss.append(tmp if tmp != np.nan and tmp != np.NAN else 0)
        mean_loss.append(np.mean(loss))
        std_loss.append(np.std(loss))

    mean_loss, std_loss = np.array(mean_loss), np.array(std_loss)
    fig.add_trace(
        go.Scatter(x=x_axis, y=mean_loss, mode="markers+lines", name="Mean Loss",
                   line=dict(dash="dash"), marker=dict(color="green", opacity=.7)))
    fig.add_trace(
        go.Scatter(x=x_axis, y=mean_loss - 2 * std_loss, fill=None, mode="lines",
                   line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(
        go.Scatter(x=x_axis, y=mean_loss + 2 * std_loss, fill='tonexty', mode="lines",
                   line=dict(color="lightgrey"), showlegend=False))

    fig.update_layout(
        title="MSE over increasing training set ",
        xaxis_title="Sample Size (in percentage)",
        yaxis_title="MSE",
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig.show()


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    feature_response_corr = get_features_to_response_correlation(X, y)

    for feature in feature_response_corr.index:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[feature],
            y=y,
            name=feature,
            mode='markers'
        ))

        fig.update_layout(
            # title=f"Feature: {feature}\nCorrelation:{feature_response_corr[feature]}",
            title=go.layout.Title(
                text=f"Feature {feature} <br><sup>"
                     f"Correlation: {feature_response_corr[feature]}</sup>",
            ),
            xaxis_title=f"{feature}",
            yaxis_title="Price",
            legend_title="Legend Title")

        fig.write_image(output_path + f"\\{feature}.jpeg")

    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset

    houses_df, prices = load_data("house_prices.csv")
    # The next line would pick the 21 most correlated features with the target variable.
    # We decided not to use this function, because the MSE is higher than if we
    # wouldn't.
    # houses_df = pick_k_most_correlated_features(houses_df, prices, 21)
    # raise NotImplementedError()

    # Question 2 - Feature evaluation with respect to response

    feature_evaluation(houses_df, prices, "ex2_plots")
    # raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.

    train_X, train_Y, test_X, test_Y = split_train_test(houses_df, prices)
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data

    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
    plot_mse(train_X, train_Y, test_X, test_Y)
