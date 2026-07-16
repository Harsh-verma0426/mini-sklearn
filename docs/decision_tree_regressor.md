# Decision Tree Regressor

## Overview

Decision Tree Regressor is a supervised machine learning algorithm used for predicting continuous numerical values. Instead of fitting a single mathematical equation, it recursively partitions the dataset into smaller regions and predicts the average target value within each region.

This implementation builds a binary regression tree using **Variance Reduction** as the splitting criterion.

---

# How It Works

The algorithm follows these steps:

1. Compute the variance of the current node.
2. Evaluate every possible split across all features.
3. Calculate the Variance Reduction for each split.
4. Select the split with the highest Variance Reduction.
5. Divide the dataset into two subsets.
6. Repeat the process recursively for each child node.
7. Stop when a leaf node is reached.

---

# Variance

Variance measures how spread out the target values are.

```text
Variance = (1/n) Σ(y - ȳ)²
```

Lower variance indicates that the samples are more similar.

---

# Variance Reduction

Variance Reduction measures how much the target variance decreases after a split.

```text
Variance Reduction =
Parent Variance
−
Weighted Child Variance
```

The weighted child variance is computed as

```text
(N_left / N) × Variance(left)
+
(N_right / N) × Variance(right)
```

A larger Variance Reduction indicates a better split.

---

# Splitting Criterion

For every feature:

1. Find all unique values.
2. Use each value as a candidate threshold.
3. Split the dataset into

```text
Left  : feature ≤ threshold
Right : feature > threshold
```

4. Compute the Variance Reduction.
5. Keep the split with the highest reduction.

---

# Stopping Criteria

Tree construction stops when:

- All target values are identical.
- No split provides positive Variance Reduction.
- A split produces an empty child node.

In these cases, the node becomes a leaf.

---

# Prediction

Prediction starts from the root node.

For each internal node:

```text
feature ≤ threshold ?
```

- Yes → Traverse left child.
- No → Traverse right child.

Repeat until a leaf node is reached.

The mean target value stored in the leaf node is returned as the prediction.

---

# Implementation Details

Current implementation includes:

- Binary regression tree
- Variance impurity measure
- Variance Reduction split criterion
- Recursive tree construction
- Binary tree traversal for prediction
- Automatic leaf node creation
- Mean target prediction

---

# Computational Complexity

| Operation | Complexity |
|----------|------------|
| Fit | O(n_features × n_samples²) *(naive implementation)* |
| Predict | O(tree depth) per sample |

---

# Example

```python
from mini_sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Comparison with scikit-learn

| Feature | This Project | scikit-learn |
|----------|--------------|--------------|
| Regression | ✅ | ✅ |
| Variance Reduction | ✅ | ✅ |
| Recursive Tree Construction | ✅ | ✅ |
| Mean Leaf Prediction | ✅ | ✅ |
| Pruning | ❌ | ✅ |
| max_depth | ❌ | ✅ |
| min_samples_split | ❌ | ✅ |
| min_samples_leaf | ❌ | ✅ |
| Feature Importance | ❌ | ✅ |

---

# Future Improvements

- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples per leaf
- Cost-complexity pruning
- Random feature selection
- Feature importance calculation
- Decision tree visualization