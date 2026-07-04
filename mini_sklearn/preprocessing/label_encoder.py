import numpy as np

from ..utils import (check_y,
                     check_is_fitted
                     )
                     
class LabelEncoder:

    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None
        self.index_to_class_ = None
        self.is_fitted_ = False

    def fit(self, X):

        X = check_y(X)

        self.classes_ = np.unique(X)
        idx = np.arange(self.classes_.size)
        self.class_to_index_ = dict(zip(self.classes_, idx))
        self.index_to_class_ = dict(zip(idx, self.classes_))

        self.is_fitted_ = True

        return self
    
    def transform(self, X):

        check_is_fitted(self.is_fitted_)
        X = check_y(X)

        X_encoded = []
        for label in X:

            if label not in self.class_to_index_:
                raise ValueError(f"Unknown label '{label}' encountered during transform.")
            
            X_encoded.append(self.class_to_index_[label])

        return np.array(X_encoded)
    
    def fit_transform(self, X):
        
        self.fit(X)

        return self.transform(X)
    
    def inverse_transform(self, X):

        check_is_fitted(self.is_fitted_)
        X = check_y(X)

        X_decoded = []
        for label in X:

            if label not in self.index_to_class_:
                raise ValueError(f"Unknown encoded label '{label}' encountered during inverse_transform().")
            
            X_decoded.append(self.index_to_class_[label])

        return np.array(X_decoded)