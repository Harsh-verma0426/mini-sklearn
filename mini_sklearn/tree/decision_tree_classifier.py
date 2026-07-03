import numpy as np
from collections import Counter
from ..utils import (
    check_X_y,
    check_X,
    check_is_fitted,
    check_feature_count,
    check_max_features
)

class Node:
  def __init__(self,
              feature = None,
              threshold = None,
              left = None,
              right = None,
              value = None,
              gain = None
          ):
    
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value
    self.gain = gain


class DecisionTreeClassifier:
  
  def __init__(self, max_features=None):
    self.max_features = max_features
    self.root = None
    self.n_features_in_ = None
    self.is_fitted_ = False

  def fit(self, X, y):

    check_X_y(X, y)
    check_max_features(self.max_features, X)

    self.root = self._build_tree(X, y)

    self.n_features_in_ = X.shape[1]
    self.is_fitted_ = True

    return self

  def predict(self, X):
    
    check_X(X)
    check_feature_count(self.n_features_in_, X)
    check_is_fitted(self.is_fitted_)

    return np.array(
      [self._predict_one(sample) for sample in X]
    )

  def _predict_one(self, sample):
    current = self.root

    while current.value is None:

        if sample[current.feature] <= current.threshold:
            current = current.left

        else:
            current = current.right

    return current.value

  def _entropy(self, y):

    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts/len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

  def _information_gain(self, y, left_idx, right_idx, parent_entropy):

    y_left = y[left_idx]
    y_right = y[right_idx]

    if len(y_left) == 0 or len(y_right) == 0:
      return 0
    
    entropy_left = self._entropy(y_left)
    entropy_right = self._entropy(y_right)

    weight_left = len(y_left)/len(y)
    weight_right = len(y_right)/len(y)

    child_entropy = (entropy_left*weight_left) + (entropy_right*weight_right)

    return parent_entropy-child_entropy

  def _split(self, X, feature, threshold):
      
      left_idx = X[:, feature] <= threshold
      right_idx = X[:, feature] > threshold

      return left_idx, right_idx

  def _find_best_split(self, X, y):

    best_feature = None
    best_threshold = None
    best_gain = -1
    parent_entropy = self._entropy(y)

    if self.max_features is None:
      features = np.arange(X.shape[1])
    else:
      features = np.random.choice(X.shape[1], size=self.max_features, replace=False)

    for feature in features:
      
      thresholds = np.unique(X[:, feature])

      for threshold in thresholds:
        left_idx, right_idx = self._split(X, feature, threshold)
        
        gain = self._information_gain(y, left_idx, right_idx, parent_entropy)

        if gain>best_gain:
          best_gain = gain
          best_feature = feature
          best_threshold = threshold

    return Node(
          feature=best_feature,
          threshold=best_threshold,
          gain=best_gain
        )

  def _build_tree(self, X, y):

    if len(np.unique(y)) == 1:
      return Node(value=y[0])
    
    node = self._find_best_split(X, y)

    if node.gain <= 0:
      return Node(value=self._most_common_label(y))
    
    left_idx, right_idx = self._split(X, node.feature, node.threshold)

    if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
      return Node(value=self._most_common_label(y))
    
    node.left = self._build_tree(X[left_idx], y[left_idx])
    node.right = self._build_tree(X[right_idx], y[right_idx])

    return node

  def _most_common_label(self, y):

    mode = Counter(y).most_common(1)[0][0]

    return mode
