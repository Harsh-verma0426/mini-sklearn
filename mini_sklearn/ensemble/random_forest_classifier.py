import numpy as np
from collections import Counter

from ..tree import DecisionTreeClassifier
from ..utils import (
    check_X_y,
    check_X,
    check_is_fitted,
    check_feature_count
)

class RandomForestClassifier:

    def __init__(self, n_estimators=100, max_features=None, max_samples=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.n_features_in_ = None
        self.is_fitted_ = False
        self.trees_ = []

    def _bootstrap_sample(self, X, y):

        original_size = len(X)

        if self.max_samples is None:
            sample_size = original_size
        else:
            sample_size = int(original_size*self.max_samples)

        final_indices = np.random.choice(original_size, size=sample_size, replace=True)

        return X[final_indices], y[final_indices]

    def fit(self, X, y):

        self.trees_ = []
        check_X_y(X, y)

        for _ in range(self.n_estimators):

            X_boot, y_boot = self._bootstrap_sample(X, y)

            tree = DecisionTreeClassifier(max_features=self.max_features)

            tree.fit(X_boot, y_boot)

            self.trees_.append(tree)
        
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def predict(self, X):

        check_X(X)
        check_feature_count(self.n_features_in_, X)
        check_is_fitted(self.is_fitted_)

        all_predictions = np.array(
            [tree.predict(X) for tree in self.trees_]
        )
        
        all_predictions = all_predictions.T

        final_predictions = []
        for votes in all_predictions:
            common_votes = Counter(votes).most_common(1)[0][0]
            final_predictions.append(common_votes)

        return np.array(final_predictions)