import numpy as np

from ..utils import (check_X,
                     check_is_fitted,
                     check_feature_count)

class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    def fit(self, X):

        X = check_X(X)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1, self.std_)

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self
    
    def transform(self, X):

        check_is_fitted(self.is_fitted_)
        X = check_X(X)
        check_feature_count(self.n_features_in_, X)

        X_scaled = (X - self.mean_) / self.std_

        return X_scaled
    
    def fit_transform(self, X):
        
        self.fit(X)

        return self.transform(X)