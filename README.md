# mini-sklearn

A lightweight Python library implementing classical machine learning algorithms from first principles using **NumPy**. The project focuses on understanding algorithmic foundations while providing a clean, **scikit-learn-inspired API**.

> **Note:** This project is intended for learning and educational purposes rather than production use.

---

## Features

### Linear Models

* ✅ Linear Regression

  * Ordinary Least Squares (OLS)
  * Gradient Descent
* ✅ Logistic Regression

### Nearest Neighbors

* ✅ K-Nearest Neighbors Classifier
* ✅ K-Nearest Neighbors Regressor

### Tree-Based Models

* ✅ Decision Tree Classifier

### Metrics

* ✅ R² Score
* ✅ Accuracy Score

### Model Selection

* ✅ Train-Test Split

---

## Documentation

Detailed implementation notes for each algorithm are available in the `docs/` directory.

* 📄 [Linear Regression](docs/linear_regression.md)
* 📄 [Logistic Regression](docs/logistic_regression.md)
* 📄 [Decision Tree Classifier](docs/decision_tree_classifier.md)
* 📄 [K-Nearest Neighbors Classifier](docs/knn_classifier.md)
* 📄 [K-Nearest Neighbors Regressor](docs/knn_regressor.md)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Harsh-verma0426/mini-sklearn.git
cd mini-sklearn
```

Create a virtual environment:

```bash
uv venv
```

Install dependencies:

```bash
uv sync
```

---

## Quick Example

```python
from mini_sklearn.linear_model import LinearRegression
from mini_sklearn.model_selection import train_test_split
from mini_sklearn.metrics import r2score

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression(method="gd")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = model.r2score(y_test, y_pred)

print(score)
```

---

## Testing

The project includes unit tests for implemented algorithms using **pytest**.

Run all tests:

```bash
python -m pytest
```

---

## Roadmap

### Core Algorithms

* [x] Linear Regression
* [x] Logistic Regression
* [x] K-Nearest Neighbors Classifier
* [x] K-Nearest Neighbors Regressor
* [x] Decision Tree Classifier
* [X] Random Forest
* [ ] Gaussian Naive Bayes
* [ ] K-Means
* [ ] Principal Component Analysis (PCA)
* [ ] Ridge Regression
* [ ] Lasso Regression
* [ ] Support Vector Machine (SVM)

### Utilities

* [x] Train-Test Split
* [X] StandardScaler
* [X] MinMaxScaler
* [X] LabelEncoder
* [ ] OneHotEncoder
* [ ] K-Fold Cross Validation
* [ ] Grid Search
* [ ] Randomized Search

### Project Improvements

* [x] Algorithm Documentation
* [x] Unit Tests
* [ ] Benchmark Against scikit-learn
* [ ] Example Notebooks
* [ ] Continuous Integration (CI)

---

## References

The implementations and concepts in this project are based on the following resources:

* *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — Aurélien Géron
* NumPy Documentation
* scikit-learn Documentation
