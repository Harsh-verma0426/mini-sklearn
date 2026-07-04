import numpy as np

from ..utils import (check_X,
                     check_is_fitted,
                     check_feature_count)

class MinMaxScaler:

    def __init__(self):
        self.data_max_ = None
        self.data_min_ = None
        self.data_range_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    def fit(self, X):

        X = check_X(X)

        self.data_max_ = np.max(X, axis=0)
        self.data_min_ = np.min(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_ = np.where(self.data_range_ == 0, 1, self.data_range_)

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self
    
    def transform(self, X):

        check_is_fitted(self.is_fitted_)
        X = check_X(X)
        check_feature_count(self.n_features_in_, X)

        X_scaled = (X - self.data_min_) / self.data_range_

        return X_scaled
    
    def fit_transform(self, X):
        
        self.fit(X)

        return self.transform(X)