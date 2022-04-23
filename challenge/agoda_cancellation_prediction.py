from typing import Tuple
from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd


def _preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Cast dates columns to be real dates
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])
    df['cancellation_datetime'] = pd.to_datetime(df['cancellation_datetime'])

    # Remove unreasonable data
    df = df[df['booking_datetime'] <= df['checkin_date']]
    df = df[df['checkout_date'] > df['checkin_date']]
    df.drop(df[~(df['booking_datetime'] < df['cancellation_datetime']) &
            (df['cancellation_datetime'] < df['checkout_date'])].index)

    # df = df[(df['booking_datetime'] < df['cancellation_datetime']) &
    #         (df['cancellation_datetime'] < df['checkout_date'])]

    # Add column indicate whether the origin country booking differ from hotel country.
    df['foreign_booking'] = df['origin_country_code'] == df[
        'hotel_country_code']

    # Cast bool features to 0/1
    df['foreign_booking'] = df['foreign_booking'].astype(int)
    df['is_user_logged_in'] = df['is_user_logged_in'].astype(int)

    # Add dummies instead of hotel id and accommodation_type_name, hotel_city_code,
    # hotel_area_code
    dummies = pd.get_dummies(df['hotel_id'])
    df = pd.concat([df, dummies], axis=1)
    dummies = pd.get_dummies(df['accommadation_type_name'])
    df = pd.concat([df, dummies], axis=1)
    dummies = pd.get_dummies(df['hotel_city_code'])
    df = pd.concat([df, dummies], axis=1)
    dummies = pd.get_dummies(df['hotel_area_code'])
    df = pd.concat([df, dummies], axis=1)

    # Cast charge_option feature
    df['charge_option'] = df['charge_option'].replace(
        {'Pay Now': 1, 'Pay Later': 2,
         'Pay at Check-in': 3})

    # date1 = pd.to_datetime("2018-12-07")
    # date2 = pd.to_datetime("2018-12-13")
    # # print(date2 > df['cancellation_datetime'])
    #
    # # df["cancellation_datetime"] = df["cancellation_datetime"].fillna(0)
    #
    # isCancelled = (df['cancellation_datetime'] >= date1) & (
    #         df['cancellation_datetime'] <= date2)
    #
    # # print(isCancelled)
    #
    # df.loc[(df['cancellation_datetime'] >= date1) & (
    #         df['cancellation_datetime'] <= date2), 'cancellation_datetime'] = True
    #
    # df.loc[~((df['cancellation_datetime'] >= date1) & (
    #         df['cancellation_datetime'] <= date2)), 'cancellation_datetime'] = False

    # df = df[df['cancellation_datetime'].between(a, b, inclusive=True)]

    # Handle dates:

    df['cancellation_year'] = pd.DatetimeIndex(
        df['cancellation_datetime']).year
    df['cancellation_day_of_year'] = df['cancellation_datetime'].dt.day_of_year

    df['booking_year'] = pd.DatetimeIndex(df['booking_datetime']).year
    df['booking_day_of_year'] = df['booking_datetime'].dt.day_of_year

    df['checkin_day_of_year'] = df['checkin_date'].dt.day_of_year

    df['checkout_day_of_year'] = df['checkout_date'].dt.day_of_year

    df["cancellation_datetime"] = df["cancellation_datetime"].fillna(0)

    df[df["cancellation_datetime"] != 0] = 1

    df['cancellation_year'] = df['cancellation_year'].replace(
        {2017: 0, 2018: 2, 2019: 1})
    df['booking_year'] = df['booking_year'].replace({2017: 0, 2018: 1})


    response = df['cancellation_datetime']

    # remove those features:
    to_drop = ["h_booking_id", "hotel_id",
               'hotel_chain_code', 'hotel_brand_code', 'hotel_live_date',
               'h_customer_id', 'customer_nationality',
               'guest_nationality_country_name', 'no_of_adults',
               'no_of_children',
               'no_of_extra_bed', 'language', 'cancellation_datetime',
               'original_payment_method', 'original_payment_type',
               'accommadation_type_name', 'hotel_city_code', 'hotel_area_code',
               'cancellation_policy_code', 'hotel_country_code',
               'origin_country_code', 'original_payment_currency']

    to_drop.extend(['booking_datetime', 'checkin_date', 'checkout_date'])
    df = df.drop(to_drop, axis=1)
    # return df, target
    return df, response


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    df = pd.read_csv(filename).drop_duplicates()
    # print(df.loc[df['cancellation_datetime'].isna(), 'cancellation_datetime'])
    df, y = _preprocess(df)
    # df, y = df.dropna(), y.dropna()
    return df, y


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(
        filename,
        index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data(
        'C:/Projects/IML/IML.HUJI/datasets/agoda_cancellation_train.csv')
    # print(df.head(100))

    train_X, train_y, test_X, test_y = split_train_test(df,
                                                        cancellation_labels)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X.to_numpy(),
                                                 train_y.to_numpy())

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
