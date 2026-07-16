import numpy as np

from ..utils import (check_y,
                     check_is_fitted
                     )
                     
class OneHotEncoder:

    def __init__(self):
        self.classes_ = []
        self.class_to_index_ = None
        self.index_to_class_ = None
        self.is_fitted_ = False

    def fit(self, X):

        X = check_y(X)

        for col in range(X.shape[1]):
            uniques = np.unique(X[:, col])
            self.classes_.append(uniques)
        
        self.is_fitted_ = True

        return self

    def transform(self, X):

        check_is_fitted(self.is_fitted_)
        X = check_y(X)

        n_samples = X.shape[0]
        X_encoded = []

        for col, cats in enumerate(self.classes_):

            self.class_to_index_ = {cat:idx for idx, cat in enumerate(cats)}
            indices = np.array([self.class_to_index_[val] for val in X[:, col]])
            sub_encoded = np.zeros((n_samples, len(cats)), dtype=int)
            sub_encoded[np.arange(n_samples), indices] = 1

            X_encoded.append(sub_encoded)

        return np.hstack(X_encoded)
    
    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)
