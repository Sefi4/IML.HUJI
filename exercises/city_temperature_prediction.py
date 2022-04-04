from IMLearn.learners.regressors import PolynomialFitting
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from IMLearn.utils.utils import split_train_test

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    temp_df = pd.read_csv(filename)
    temp_df['Date'] = pd.to_datetime(temp_df.Date)
    temp_df['DayOfYear'] = temp_df['Date'].dt.dayofyear
    return temp_df[temp_df.Temp > -70]
    # raise NotImplementedError()


def plot_temp_mean_over_day_of_year(il_temp_df: pd.DataFrame):
    """
    Plot a scatter graph of the mean temperature in Israel as a function of the day in
    the year.
    ----------
    il_temp_df: pd.DataFrame
        Israel subset data sample.
    """
    il_temp_df['Year'] = il_temp_df['Year'].astype(str)
    fig = px.scatter(il_temp_df, x='DayOfYear', y='Temp', color='Year',
                     title="average temperature as a function of day of the year".capitalize())
    il_temp_df['Year'] = il_temp_df['Year'].astype(int)
    fig.show()


def plot_temperature_std_by_month(il_temp_df: pd.DataFrame):
    """
    Plot bar graph of the standard deviation of the temperature in Israel over a month.
    ----------
    il_temp_df: pd.DataFrame
        Israel subset data sample.
    """
    df = il_temp_df.groupby('Month').Temp.agg('std')
    fig = px.bar(df, title='STD Of Temperature over Month')
    fig.show()


def plot_average_monthly_temperature(temp_df: pd.DataFrame):
    """
    Plot the mean monthly temperature of all countries in the data set.
    ----------
    temp_df: pd.DataFrame
        The loaded data set.
    """
    df = temp_df.groupby(['Country', 'Month']).Temp.agg(['mean', 'std'])
    fig = px.line(df, x=df.index.get_level_values('Month'), y='mean',
                  color=df.index.get_level_values('Country'),
                  title='Average Monthly Temperature',
                  error_y='std')
    fig.show()


def plot_mse_over_israel(il_temp_df: pd.DataFrame):
    """
    Plot bar graph of the MSE of the temperature in Israel as a function of the day of
    the year. Each bar represent the MSE calculated under different polynomial degree.
    ----------
    il_temp_df: pd.DataFrame
        Israel subset data sample.
    """
    X, y = il_temp_df['DayOfYear'], il_temp_df['Temp']
    trainX, trainY, testX, testY = split_train_test(X, y)
    trainX, trainY, testX, testY = trainX.to_numpy(), trainY.to_numpy(), \
                                   testX.to_numpy(), testY.to_numpy()
    y_axis = []
    for k in range(1, 11):
        pl = PolynomialFitting(k)
        pl.fit(trainX, trainY)
        y_axis.append(round(pl.loss(testX, testY), 2))
        print("Error for k = ", k, 'is', y_axis[-1])
    fig = px.bar(x=np.linspace(1, 10, 10), y=y_axis,
                 title="The MSE For Predicting Temperature",
                 labels={
                     'x': 'Polynomial Degree',
                     'y': 'MSE'})
    fig.show()


def plot_mse(temp_df: pd.DataFrame):
    """
    Plot bar graph of the MSE of the temperature in each country as a function of the
    day of the year. Each bar represent the MSE calculated for each country.
    ----------
    il_temp_df: pd.DataFrame
        Israel subset data sample.
    """
    X, y = il_temp_df['DayOfYear'], il_temp_df['Temp']
    pl = PolynomialFitting(5)
    pl.fit(X, y)
    y_axis = []

    countries = np.array(['Jordan', 'South Africa', 'The Netherlands'])
    for country in countries:
        df = temp_df[temp_df.Country == country]
        y_axis.append(pl.loss(df.DayOfYear, df.Temp))

    fig = px.bar(x=countries, y=y_axis,
                 title="The MSE For Predicting Temperature",
                 labels={
                     'x': 'Country',
                     'y': 'MSE'})
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temp_df = load_data('City_Temperature.csv')
    # raise NotImplementedError()

    # Question 2 - Exploring data for specific country

    # Get only Israel dataFrame and remove unreasonable temperature
    il_temp_df = temp_df.loc[temp_df.Country == 'Israel'].copy()
    # looks like a 4 deg polynomial.
    plot_temp_mean_over_day_of_year(il_temp_df)
    plot_temperature_std_by_month(il_temp_df)

    # raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    plot_average_monthly_temperature(temp_df)

    # raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    plot_mse_over_israel(il_temp_df)

    # raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    plot_mse(temp_df)
    # raise NotImplementedError()
