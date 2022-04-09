from typing import Tuple

from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd


def _preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = df['cancellation_policy_code']

    # Cast dates columns to be real dates
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime']).dt.date
    df['checkin_date'] = pd.to_datetime(df['checkin_date']).dt.date
    df['checkout_date'] = pd.to_datetime(df['checkout_date']).dt.date
    df['cancellation_datetime'] = pd.to_datetime(df['cancellation_datetime']).dt.date

    # Remove unreasonable data
    df = df[df['booking_datetime'] <= df['checkin_date']]
    df = df[df['checkout_date'] > df['checkin_date']]
    df = df[(df['booking_datetime'] < df['cancellation_datetime']) &
            (df['cancellation_datetime'] < df['checkout_date'])]

    # Add column indicate whether the origin country booking differ from hotel country.
    df['foreign_book'] = df['origin_country_code'] == df['hotel_country_code']

    # Cast bool features to 0/1
    df['foreign_book'] = df['foreign_book'].astype(int)
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
    df['charge_option'] = df['charge_option'].replace({'Pay Now': 1, 'Pay Later': 2})

    # remove those features:
    to_drop = ["h_booking_id", "hotel_id",
               'hotel_chain_code', 'hotel_brand_code', 'hotel_live_date',
               'h_customer_id', 'customer_nationality',
               'guest_nationality_country_name', 'no_of_adults', 'no_of_children',
               'no_of_extra_bed', 'language', 'cancellation_datetime',
               'original_payment_method', 'original_payment_type',
               'accommadation_type_name', 'hotel_city_code', 'hotel_area_code']

    df = df.drop(to_drop, axis=1).reset_index()


    return df, target


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
    df = pd.read_csv(filename).dropna().drop_duplicates()
    return _preprocess(df)


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
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
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename,
                                                                            index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    print(df.head(100))

    # train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
