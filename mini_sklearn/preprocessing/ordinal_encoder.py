import numpy as np

from ..utils import (check_y,
                     check_is_fitted
                     )
                     
class OrdinalEncoder:

    def __init__(self, categories):
        self.classes_ = categories
        self.class_to_index_ = None
        self.is_fitted_ = False

    def fit(self, X):

        X = check_y(X)
        self.class_to_index_ = [{cat: idx for idx, cat in enumerate(cats)} for cats in self.classes_]

        self.is_fitted_ = True

        return self
    
    def transform(self, X):

        check_is_fitted(self.is_fitted_)
        X = check_y(X)

        X_encoded = []
        for col in range(len(self.class_to_index_)):
            feature_map = self.class_to_index_[col]

            lookup_func = np.vectorize(lambda label: feature_map.get(label, -1))

            sub_encoded = lookup_func(X[:, col])  

            X_encoded.append(sub_encoded)
        
        return np.column_stack(X_encoded)
    
    def fit_transform(self, X):
        
        self.fit(X)

        return self.transform(X)
    