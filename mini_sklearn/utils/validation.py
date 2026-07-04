import numpy as np
    
def check_numeric(array):
    # 'i' = integer, 'f' = float, 'u' = unsigned int
    if array.dtype.kind not in ('i', 'f', 'u'):
        raise TypeError(f"Expected numeric data, but got text/object data type: {array.dtype}")

def convert_into_array(array):
    
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    return array

def make_2D(X):

    if X.ndim == 1:
            X = X.reshape(-1, 1)

    return X

def check_1d_array(array):

    if array.ndim != 1:
        raise ValueError(f"Expected 1D array, got {array.ndim}D array instead")

def check_same_number_of_samples(X, y):

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: {[X.shape[0], y.shape[0]]}")
    
def check_not_empty(array):

    if array.size == 0:
        raise ValueError("Array is empty")
    
def check_X_y(X, y):

    X = convert_into_array(X)
    y = convert_into_array(y)

    check_not_empty(X)
    check_not_empty(y)
    check_numeric(X)
    check_numeric(y)

    X = make_2D(X)

    check_1d_array(y)
    check_same_number_of_samples(X, y)

    return X, y

def check_X(X):

    X = convert_into_array(X)

    check_not_empty(X)
    check_numeric(X)

    X = make_2D(X)

    return X
    
def check_is_fitted(is_fitted):

    if not is_fitted:
        raise ValueError("This estimator is not fitted yet. Call 'fit' before using this estimator.")
    
def check_feature_count(features, X):

    if features != X.shape[1]:
        raise ValueError(
            f"X has {X.shape[1]} features, but the model was fitted with {features} features."
        )
    
def check_max_features(max_features, X):

    if max_features is not None and max_features <= 0:
        raise ValueError(
            "max_features must be a positive integer."
        )

    if max_features is not None and max_features > X.shape[1]:
        raise ValueError(
            f"max_features={max_features} is greater than the number of features ({X.shape[1]})."
        )