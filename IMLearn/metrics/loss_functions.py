import numpy as np
import sklearn.metrics


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    # raise NotImplementedError()
    assert y_true.size > 0
    assert not np.any(np.isnan(y_true))
    assert not np.any(np.isnan(y_pred))
    return sum(((y_true - y_pred) ** 2) / y_true.size)


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    n = y_true.size
    res = np.sum(y_true * y_pred <= 0)
    return res if not normalize else res / n
    # raise NotImplementedError()


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    raise NotImplementedError()


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()


if __name__ == '__main__':
    y_true = np.array([1, 1, 1, 1, 1, 1])
    # y_pred = np.array([1, 1, -1, 1, -1, 1])
    # print(misclassification_error(y_true, y_pred, False))
    print(np.argmax(y_true < 0))
