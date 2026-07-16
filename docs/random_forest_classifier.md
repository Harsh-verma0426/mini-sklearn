# Random Forest Classifier

## Overview

Random Forest Classifier is a supervised ensemble machine learning algorithm used for classification tasks. Instead of training a single Decision Tree, it builds multiple Decision Trees on different bootstrap samples of the training data and combines their predictions using majority voting.

This implementation uses **Bootstrap Aggregating (Bagging)** and **Decision Trees** as base learners.

---

# How It Works

The algorithm follows these steps:

1. Generate multiple bootstrap samples from the training dataset.
2. Train one Decision Tree on each bootstrap sample.
3. Repeat until all trees are trained.
4. During prediction, obtain predictions from every tree.
5. Use majority voting to determine the final class.

---

# Bootstrap Sampling

Each Decision Tree is trained on a randomly sampled dataset generated **with replacement**.

Because sampling is performed with replacement:

- Some training samples may appear multiple times.
- Some samples may not appear at all.

This introduces diversity among the trees and helps reduce overfitting.

---

# Majority Voting

Each Decision Tree predicts a class label.

Example:

```text
Tree 1 → Class 0
Tree 2 → Class 1
Tree 3 → Class 1
Tree 4 → Class 0
Tree 5 → Class 1
```

Prediction:

```text
Class 1
```

The class receiving the highest number of votes is returned as the final prediction.

---

# Training

For each Decision Tree:

1. Generate a bootstrap sample.
2. Train a Decision Tree using the sampled data.
3. Store the trained tree.

Repeat until all trees have been trained.

---

# Prediction

For every query sample:

1. Pass the sample through every Decision Tree.
2. Collect predictions from all trees.
3. Count the frequency of each predicted class.
4. Return the class with the highest frequency.

---

# Implementation Details

Current implementation includes:

- Bootstrap sampling
- Multiple Decision Trees
- Majority voting
- Binary and multiclass classification support
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
from mini_sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Comparison with scikit-learn

| Feature | This Project | scikit-learn |
|----------|--------------|--------------|
| Classification | ✅ | ✅ |
| Bootstrap Sampling | ✅ | ✅ |
| Majority Voting | ✅ | ✅ |
| Binary Classification | ✅ | ✅ |
| Multiclass Classification | ✅ | ✅ |
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
- Extra Trees implementation
```