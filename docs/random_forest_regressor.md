# Random Forest Regressor

## Overview

Random Forest Regressor is a supervised ensemble machine learning algorithm used for regression tasks. Instead of training a single Decision Tree, it builds multiple Decision Trees on different bootstrap samples of the training data and combines their predictions by averaging the outputs.

This implementation uses **Bootstrap Aggregating (Bagging)** and **Decision Tree Regressors** as base learners.

---

# How It Works

The algorithm follows these steps:

1. Generate multiple bootstrap samples from the training dataset.
2. Train one Decision Tree Regressor on each bootstrap sample.
3. Repeat until all trees have been trained.
4. During prediction, obtain predictions from every tree.
5. Compute the average of all predictions.

---

# Bootstrap Sampling

Each Decision Tree is trained on a randomly sampled dataset generated **with replacement**.

Because sampling is performed with replacement:

- Some training samples may appear multiple times.
- Some samples may not appear at all.

This introduces diversity among the trees and reduces overfitting.

---

# Mean Prediction

Each Decision Tree predicts a continuous value.

Example:

```text
Tree 1 → 10.5
Tree 2 → 11.2
Tree 3 → 10.8
Tree 4 → 11.5
Tree 5 → 11.0
```

Prediction:

```text
(10.5 + 11.2 + 10.8 + 11.5 + 11.0) / 5 = 11.0
```

The arithmetic mean of all tree predictions is returned as the final prediction.

---

# Training

For each Decision Tree:

1. Generate a bootstrap sample.
2. Train a Decision Tree Regressor using the sampled data.
3. Store the trained tree.

Repeat until all trees have been trained.

---

# Prediction

For every query sample:

1. Pass the sample through every Decision Tree Regressor.
2. Collect predictions from all trees.
3. Compute the arithmetic mean of the predictions.
4. Return the mean as the final prediction.

---

# Implementation Details

Current implementation includes:

- Bootstrap sampling
- Multiple Decision Tree Regressors
- Mean prediction
- Continuous target prediction
- Configurable number of estimators

---

# Computational Complexity

| Operation | Complexity |
|----------|------------|
| Fit | O(n_estimators × Tree Training Time) |
| Predict | O(n_estimators × Tree Prediction Time) |

---

# Example

```python
from mini_sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Comparison with scikit-learn

| Feature | This Project | scikit-learn |
|----------|--------------|--------------|
| Regression | ✅ | ✅ |
| Bootstrap Sampling | ✅ | ✅ |
| Mean Prediction | ✅ | ✅ |
| Multiple Decision Trees | ✅ | ✅ |
| Feature Importance | ❌ | ✅ |
| Out-of-Bag (OOB) Score | ❌ | ✅ |
| Parallel Training | ❌ | ✅ |
| Random Feature Selection | ❌ | ✅ |
| max_depth | ❌ | ✅ |
| min_samples_split | ❌ | ✅ |

---

# Future Improvements

- Random feature selection at each split
- Feature importance calculation
- Out-of-Bag (OOB) score
- Parallel tree training
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples per leaf
- Extra Trees Regressor
```