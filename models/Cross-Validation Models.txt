# Cross-Validation Models — Full Notes

## Beginner → Intermediate → Advanced

### Models Covered

```text
RidgeCV
LassoCV
ElasticNetCV
MultiTaskLassoCV
MultiTaskElasticNetCV
LogisticRegressionCV
```

These models combine **machine learning algorithms with Cross-Validation (CV)** to automatically select important hyperparameters.

---

# 1. What is Cross-Validation?

**Cross-Validation (CV)** is a technique used to evaluate a machine learning model and select suitable hyperparameters.

Instead of training and testing the model only once, Cross-Validation divides the training data into multiple parts called **folds**.

Example:

```text
Dataset
   |
   +---- Fold 1
   +---- Fold 2
   +---- Fold 3
   +---- Fold 4
   +---- Fold 5
```

For 5-Fold Cross-Validation:

```text
Round 1:
Train → Fold 2,3,4,5
Validation → Fold 1

Round 2:
Train → Fold 1,3,4,5
Validation → Fold 2

Round 3:
Train → Fold 1,2,4,5
Validation → Fold 3

Round 4:
Train → Fold 1,2,3,5
Validation → Fold 4

Round 5:
Train → Fold 1,2,3,4
Validation → Fold 5
```

Finally, the validation scores are averaged.

---

# 2. Why Do We Need Cross-Validation?

Suppose we want to choose:

```text
alpha = 0.01
alpha = 0.1
alpha = 1
alpha = 10
alpha = 100
```

Which one is best?

Instead of manually selecting one, Cross-Validation tests different values and selects a suitable value based on validation performance.

Therefore:

```text
Cross-Validation
       ↓
Test multiple hyperparameters
       ↓
Evaluate each using CV
       ↓
Select the best value
       ↓
Train final model
```

---

# 3. What Does CV Mean in Model Names?

Consider:

```python
RidgeCV
```

The `CV` means:

```text
Cross-Validation
```

Similarly:

```text
LassoCV
ElasticNetCV
MultiTaskLassoCV
MultiTaskElasticNetCV
LogisticRegressionCV
```

These models perform some form of cross-validation internally.

---

# 4. Regularization and Cross-Validation

Most models in this section use **regularization**.

Regularization helps control model complexity.

Common regularization methods:

```text
L1 Regularization → Lasso
L2 Regularization → Ridge
L1 + L2 → ElasticNet
```

Cross-validation helps choose the strength of regularization.

For example:

```text
Ridge
   |
   +---- alpha = 0.01
   +---- alpha = 0.1
   +---- alpha = 1
   +---- alpha = 10
   +---- alpha = 100
             ↓
      Cross-Validation
             ↓
       Best alpha
```

---

# 5. RidgeCV

## 5.1 What is RidgeCV?

`RidgeCV` is a Ridge Regression model that automatically selects the best regularization parameter `alpha` using Cross-Validation.

```python
from sklearn.linear_model import RidgeCV
```

Basic syntax:

```python
RidgeCV(alphas=(0.1, 1.0, 10.0))
```

---

# 6. Ridge Regression

Before understanding RidgeCV, understand Ridge Regression.

Ridge Regression uses **L2 regularization**.

Objective function:

```text
Loss = MSE + α Σ w²
```

More formally:

```text
minimize:

Σ(yᵢ - ŷᵢ)² + α Σwⱼ²
```

Where:

```text
α = regularization strength
w = model coefficients
```

---

# 7. How RidgeCV Works

Suppose:

```python
alphas = [0.01, 0.1, 1, 10, 100]
```

RidgeCV roughly performs:

```text
alpha = 0.01 → CV score
alpha = 0.1  → CV score
alpha = 1    → CV score
alpha = 10   → CV score
alpha = 100  → CV score
```

Then:

```text
Best CV performance
       ↓
Best alpha
```

Example:

```python
from sklearn.linear_model import RidgeCV

model = RidgeCV(
    alphas=[0.01, 0.1, 1, 10, 100]
)

model.fit(X_train, y_train)

print(model.alpha_)
```

---

# 8. RidgeCV Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RidgeCV(
    alphas=[0.01, 0.1, 1, 10, 100]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Best alpha:", model.alpha_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

# 9. Important RidgeCV Attributes

```python
model.alpha_
```

Returns the selected alpha.

Other useful attributes:

```python
model.coef_
model.intercept_
```

Example:

```python
print(model.alpha_)
print(model.coef_)
print(model.intercept_)
```

---

# 10. Important RidgeCV Parameters

Common parameters:

```text
alphas
fit_intercept
scoring
cv
gcv_mode
store_cv_results
alpha_per_target
```

### `alphas`

Values of alpha to test.

```python
RidgeCV(
    alphas=[0.01, 0.1, 1, 10, 100]
)
```

### `cv`

Controls cross-validation.

```python
RidgeCV(
    alphas=[0.1, 1, 10],
    cv=5
)
```

### `scoring`

Controls evaluation metric.

```python
RidgeCV(
    alphas=[0.1, 1, 10],
    scoring="neg_mean_squared_error"
)
```

---

# 11. RidgeCV Use Cases

RidgeCV is useful when:

* Features are highly correlated
* You have many features
* You want L2 regularization
* You don't want to manually choose alpha
* You want automatic hyperparameter selection

---

# 12. LassoCV

## 12.1 What is LassoCV?

`LassoCV` is Lasso Regression with built-in Cross-Validation.

```python
from sklearn.linear_model import LassoCV
```

It automatically selects the best `alpha`.

---

# 13. Lasso Regression

Lasso uses **L1 regularization**.

Objective:

```text
Loss = MSE + α Σ|w|
```

More formally:

```text
minimize:

Σ(yᵢ - ŷᵢ)² + α Σ|wⱼ|
```

The important feature of Lasso:

```text
Some coefficients can become exactly 0
```

Therefore Lasso can perform **feature selection**.

---

# 14. Ridge vs Lasso

```text
Ridge
L2
Σw²
Usually shrinks coefficients toward 0

Lasso
L1
Σ|w|
Can make coefficients exactly 0
```

Therefore:

```text
Ridge → coefficient shrinkage

Lasso → coefficient shrinkage + feature selection
```

---

# 15. LassoCV Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LassoCV(
    alphas=[0.001, 0.01, 0.1, 1, 10],
    cv=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Best alpha:", model.alpha_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

# 16. LassoCV Feature Selection Example

```python
from sklearn.linear_model import LassoCV

model = LassoCV(cv=5)

model.fit(X_train, y_train)

print("Best alpha:", model.alpha_)
print("Coefficients:", model.coef_)
```

Find zero coefficients:

```python
import numpy as np

zero_features = np.sum(model.coef_ == 0)

print("Zero coefficients:", zero_features)
```

If a coefficient is:

```text
0
```

the corresponding feature has effectively been removed from the linear model.

---

# 17. Important LassoCV Parameters

Common parameters:

```text
alphas
cv
max_iter
tol
selection
random_state
n_jobs
```

Example:

```python
model = LassoCV(
    alphas=[0.001, 0.01, 0.1, 1],
    cv=5,
    max_iter=10000
)
```

---

# 18. When Should You Use LassoCV?

Use LassoCV when:

* You have many features
* Some features may be irrelevant
* Feature selection is important
* You want L1 regularization
* You don't know the best alpha

---

# 19. ElasticNetCV

## 19.1 What is ElasticNetCV?

`ElasticNetCV` combines:

```text
L1 Regularization
+
L2 Regularization
```

and uses Cross-Validation to select suitable hyperparameters.

```python
from sklearn.linear_model import ElasticNetCV
```

---

# 20. ElasticNet

ElasticNet combines Lasso and Ridge.

Conceptually:

```text
ElasticNet
=
L1 penalty
+
L2 penalty
```

Objective:

```text
MSE
+
α × l1_ratio × L1
+
α × (1-l1_ratio) × L2
```

Where:

```text
alpha = overall regularization strength

l1_ratio = balance between L1 and L2
```

---

# 21. Understanding `l1_ratio`

Suppose:

```python
l1_ratio = 1
```

Then:

```text
100% L1
```

This behaves like Lasso.

If:

```python
l1_ratio = 0
```

then:

```text
100% L2
```

This behaves conceptually like Ridge.

Examples:

```text
l1_ratio = 0.1
→ mostly L2

l1_ratio = 0.5
→ balanced L1 + L2

l1_ratio = 0.9
→ mostly L1
```

---

# 22. How ElasticNetCV Selects Parameters

ElasticNetCV can test:

```text
alpha values
+
l1_ratio values
```

For example:

```text
alpha:
0.01
0.1
1
10

l1_ratio:
0.1
0.5
0.9
```

It evaluates combinations:

```text
alpha=0.01, l1_ratio=0.1
alpha=0.01, l1_ratio=0.5
alpha=0.01, l1_ratio=0.9

alpha=0.1, l1_ratio=0.1
...
```

Then selects the best combination.

---

# 23. ElasticNetCV Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = ElasticNetCV(
    alphas=[0.001, 0.01, 0.1, 1],
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Best alpha:", model.alpha_)
print("Best l1_ratio:", model.l1_ratio_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

# 24. Important ElasticNetCV Attributes

```python
model.alpha_
```

Best alpha.

```python
model.l1_ratio_
```

Best L1/L2 balance.

```python
model.coef_
```

Learned coefficients.

```python
model.intercept_
```

Intercept.

---

# 25. When Should You Use ElasticNetCV?

ElasticNetCV is especially useful when:

* You have many features
* Features are correlated
* You want feature selection
* Pure Lasso is unstable
* You want both L1 and L2 regularization

---

# 26. Ridge vs Lasso vs ElasticNet

| Model      | Regularization | Feature Selection     | CV               |
| ---------- | -------------- | --------------------- | ---------------- |
| Ridge      | L2             | Usually no exact zero | RidgeCV yes      |
| Lasso      | L1             | Yes                   | LassoCV yes      |
| ElasticNet | L1 + L2        | Yes                   | ElasticNetCV yes |

---

# 27. MultiTask Learning

Before understanding MultiTaskLassoCV and MultiTaskElasticNetCV, we need to understand **Multi-Output Regression**.

Suppose we have:

```text
X
 ↓
Model
 ↓
y1
y2
y3
```

There are multiple target variables.

Example:

```text
Input:
house size, rooms, location

Outputs:
house price
rent price
tax estimate
```

This is a multi-output problem.

---

# 28. Single-Task vs Multi-Task

### Single-Task

```text
X → y
```

Example:

```text
X → house price
```

### Multi-Task

```text
X → y1
  → y2
  → y3
```

The tasks are related.

Multi-task models try to learn these tasks jointly.

---

# 29. MultiTaskLassoCV

## 29.1 What is MultiTaskLassoCV?

`MultiTaskLassoCV` is a multi-output linear regression model using L1-style group regularization and Cross-Validation.

```python
from sklearn.linear_model import MultiTaskLassoCV
```

It is designed for:

```text
Multiple target variables
```

---

# 30. Why MultiTaskLasso?

Suppose:

```text
X → Target 1
X → Target 2
X → Target 3
```

Some features may be useful across several tasks.

MultiTaskLasso encourages a feature to be selected or removed **across tasks together**.

Conceptually:

```text
Feature 1 → Task 1 ✓
Feature 1 → Task 2 ✓
Feature 1 → Task 3 ✓

Feature 2 → Task 1 ×
Feature 2 → Task 2 ×
Feature 2 → Task 3 ×
```

This shared feature selection is one of the important differences from applying ordinary Lasso independently to each target.

---

# 31. MultiTaskLassoCV Example

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error

X, y = make_regression(
    n_samples=300,
    n_features=20,
    n_targets=3,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultiTaskLassoCV(
    cv=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Best alpha:", model.alpha_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("Coefficient shape:", model.coef_.shape)
```

---

# 32. MultiTaskLassoCV Coefficient Structure

For multi-task models:

```python
model.coef_
```

contains coefficients for multiple targets.

Conceptually:

```text
                 Task 1   Task 2   Task 3
Feature 1          w11      w12      w13
Feature 2          w21      w22      w23
Feature 3          w31      w32      w33
```

---

# 33. Important MultiTaskLassoCV Parameters

Common parameters:

```text
alphas
cv
max_iter
tol
selection
n_jobs
random_state
```

Example:

```python
model = MultiTaskLassoCV(
    cv=5,
    max_iter=10000
)
```

---

# 34. MultiTaskLassoCV vs LassoCV

### LassoCV

Usually handles:

```text
X → one target
```

### MultiTaskLassoCV

Handles:

```text
X → multiple related targets
```

Main difference:

```text
LassoCV
→ each target independently

MultiTaskLassoCV
→ targets learned jointly
```

---

# 35. MultiTaskElasticNetCV

## 35.1 What is MultiTaskElasticNetCV?

`MultiTaskElasticNetCV` is a multi-output regression model combining:

```text
L1 regularization
+
L2 regularization
+
Cross-Validation
```

It is useful when there are multiple related regression targets.

```python
from sklearn.linear_model import MultiTaskElasticNetCV
```

---

# 36. MultiTaskElasticNetCV Concept

Suppose:

```text
X
 ↓
 ├── Target 1
 ├── Target 2
 └── Target 3
```

The model learns all targets together.

It uses:

```text
alpha
+
l1_ratio
```

and selects suitable values using Cross-Validation.

---

# 37. MultiTaskElasticNetCV Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.metrics import mean_squared_error

X, y = make_regression(
    n_samples=300,
    n_features=20,
    n_targets=3,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultiTaskElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5,
    max_iter=10000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Best alpha:", model.alpha_)
print("Best l1_ratio:", model.l1_ratio_)
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

# 38. MultiTaskElasticNetCV Important Attributes

```python
model.alpha_
```

Best regularization strength.

```python
model.l1_ratio_
```

Best L1/L2 balance.

```python
model.coef_
```

Learned coefficients.

```python
model.intercept_
```

Intercepts for multiple targets.

---

# 39. When Should You Use MultiTaskElasticNetCV?

Use it when:

* There are multiple regression targets
* Targets are related
* Features may be shared between tasks
* Feature selection is useful
* You want both L1 and L2 regularization
* You want automatic hyperparameter selection

---

# 40. LogisticRegressionCV

## 40.1 What is LogisticRegressionCV?

`LogisticRegressionCV` is Logistic Regression with built-in Cross-Validation.

```python
from sklearn.linear_model import LogisticRegressionCV
```

It automatically searches for suitable regularization parameters.

---

# 41. Logistic Regression Review

Logistic Regression is a classification algorithm.

Linear score:

```text
z = wᵀx + b
```

Sigmoid:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

Probability:

```text
P(y=1|x) = σ(z)
```

Classification:

```text
P >= 0.5 → class 1
P < 0.5  → class 0
```

---

# 42. Why LogisticRegressionCV?

Normal Logistic Regression:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0)
```

You manually select `C`.

With:

```python
from sklearn.linear_model import LogisticRegressionCV
```

Cross-validation can search over multiple `C` values.

---

# 43. Understanding `C`

For Logistic Regression:

```text
C = inverse of regularization strength
```

Therefore:

```text
Small C
→ Strong regularization

Large C
→ Weak regularization
```

This is opposite to the usual interpretation of `alpha`.

Compare:

```text
Ridge/Lasso:
larger alpha → stronger regularization

LogisticRegression:
smaller C → stronger regularization
```

---

# 44. LogisticRegressionCV Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegressionCV(
        Cs=10,
        cv=5,
        max_iter=5000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

# 45. Finding the Best C

Because the classifier is inside a Pipeline:

```python
classifier = model.named_steps["classifier"]

print(classifier.C_)
```

`C_` contains the selected regularization parameter.

For multiclass problems, `C_` may contain a value for each class depending on the configuration.

---

# 46. LogisticRegressionCV Parameters

Important parameters:

```text
Cs
fit_intercept
cv
penalty
solver
scoring
max_iter
class_weight
refit
n_jobs
random_state
```

Example:

```python
model = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1, 10, 100],
    cv=5,
    scoring="accuracy",
    max_iter=5000
)
```

---

# 47. `Cs` Parameter

`Cs` specifies candidate values for `C`.

Example:

```python
Cs=[0.01, 0.1, 1, 10, 100]
```

The model evaluates different regularization strengths.

Conceptually:

```text
C = 0.01 → CV score
C = 0.1  → CV score
C = 1    → CV score
C = 10   → CV score
C = 100  → CV score
```

Then the best value is selected.

---

# 48. LogisticRegressionCV Probability Prediction

Unlike most models in this section, LogisticRegressionCV supports:

```python
predict_proba()
```

Example:

```python
probabilities = model.predict_proba(X_test)

print(probabilities[:5])
```

For binary classification:

```text
[class 0 probability, class 1 probability]
```

---

# 49. LogisticRegressionCV Decision Function

It also provides:

```python
decision_function()
```

Example:

```python
scores = model.decision_function(X_test)

print(scores[:5])
```

The decision score is not the same thing as a probability.

---

# 50. Cross-Validation Models Summary

```text
RidgeCV
    ↓
Ridge + CV
    ↓
L2 regularization

LassoCV
    ↓
Lasso + CV
    ↓
L1 regularization

ElasticNetCV
    ↓
ElasticNet + CV
    ↓
L1 + L2

MultiTaskLassoCV
    ↓
Multi-output Lasso + CV

MultiTaskElasticNetCV
    ↓
Multi-output ElasticNet + CV

LogisticRegressionCV
    ↓
Logistic Regression + CV
    ↓
Classification
```

---

# 51. Alpha vs C

This is a very important interview concept.

### Ridge/Lasso/ElasticNet

Use:

```text
alpha
```

Larger:

```text
alpha ↑
→ regularization ↑
```

### LogisticRegressionCV

Uses:

```text
C
```

Larger:

```text
C ↑
→ regularization ↓
```

Therefore:

```text
alpha ↑ → stronger regularization

C ↑ → weaker regularization
```

---

# 52. RidgeCV vs LassoCV

| Feature                     | RidgeCV          | LassoCV          |
| --------------------------- | ---------------- | ---------------- |
| Algorithm                   | Ridge Regression | Lasso Regression |
| Regularization              | L2               | L1               |
| Parameter                   | alpha            | alpha            |
| Feature selection           | Usually no       | Yes              |
| Handles correlated features | Good             | Can be unstable  |
| Cross-Validation            | Yes              | Yes              |
| Target                      | Regression       | Regression       |

---

# 53. LassoCV vs ElasticNetCV

| Feature             | LassoCV      | ElasticNetCV     |
| ------------------- | ------------ | ---------------- |
| L1                  | Yes          | Yes              |
| L2                  | No           | Yes              |
| Feature selection   | Yes          | Yes              |
| Parameters          | alpha        | alpha + l1_ratio |
| Correlated features | Can struggle | Often better     |
| CV                  | Yes          | Yes              |

---

# 54. RidgeCV vs ElasticNetCV

```text
RidgeCV
→ only L2

ElasticNetCV
→ L1 + L2
```

Ridge generally keeps all features but shrinks coefficients.

ElasticNet can produce zero coefficients and perform feature selection.

---

# 55. LassoCV vs MultiTaskLassoCV

```text
LassoCV
→ Single-output regression

MultiTaskLassoCV
→ Multi-output regression
```

Example:

```text
LassoCV:

X → y


MultiTaskLassoCV:

X → y1
  → y2
  → y3
```

---

# 56. ElasticNetCV vs MultiTaskElasticNetCV

```text
ElasticNetCV
→ single-output regression

MultiTaskElasticNetCV
→ multiple related outputs
```

---

# 57. LogisticRegressionCV vs RidgeCV

These models are fundamentally different.

### RidgeCV

```text
Regression
↓
Continuous output
```

Example:

```text
House price = 250000
```

### LogisticRegressionCV

```text
Classification
↓
Class probability / class label
```

Example:

```text
Spam = 1
Not Spam = 0
```

---

# 58. Cross-Validation Workflow

A typical workflow:

```text
Raw Dataset
     ↓
Train/Test Split
     ↓
Training Data
     ↓
Cross-Validation
     ↓
Try Hyperparameters
     ↓
Select Best Hyperparameter
     ↓
Fit Final Model
     ↓
Test Set
     ↓
Evaluation
```

Important:

```text
Test set should not be used to select hyperparameters.
```

---

# 59. Why Keep a Separate Test Set?

Suppose:

```text
Dataset
   ↓
80% Training
20% Testing
```

Cross-validation happens inside the training data:

```text
Training Data
     ↓
5-Fold CV
     ↓
Hyperparameter selection
```

The test set remains untouched until final evaluation.

This gives a better estimate of generalization performance.

---

# 60. Cross-Validation and Data Leakage

Incorrect:

```text
Scale entire dataset
      ↓
Cross-Validation
```

The scaler has seen validation information.

Better:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegressionCV(cv=5))
])
```

Now preprocessing happens correctly within the CV training process.

---

# 61. Why Scaling Matters

Scaling is particularly important for:

```text
LassoCV
ElasticNetCV
LogisticRegressionCV
```

and generally useful for regularized linear models.

Example:

```text
Feature 1 → 0 to 1
Feature 2 → 0 to 1,000,000
```

The optimization process may become difficult or inefficient.

Use:

```python
StandardScaler()
```

Example:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoCV())
])
```

---

# 62. StandardScaler Formula

For a feature:

```text
z = (x - μ) / σ
```

Where:

```text
μ = mean
σ = standard deviation
```

After scaling, features are approximately centered around:

```text
mean = 0
standard deviation = 1
```

---

# 63. Cross-Validation Types

Common methods include:

```text
K-Fold Cross-Validation
Stratified K-Fold
Leave-One-Out
Repeated K-Fold
Time Series Split
```

For regression:

```text
KFold
```

is commonly used.

For classification:

```text
StratifiedKFold
```

is often appropriate because it preserves class proportions.

---

# 64. K-Fold Cross-Validation

Suppose:

```text
cv=5
```

Then:

```text
Fold 1 → validation
Fold 2 → validation
Fold 3 → validation
Fold 4 → validation
Fold 5 → validation
```

Every sample gets used for validation once and training multiple times.

---

# 65. Cross-Validation Score

Suppose 5 folds produce:

```text
Fold 1 = 0.90
Fold 2 = 0.87
Fold 3 = 0.91
Fold 4 = 0.89
Fold 5 = 0.88
```

Average:

```text
(0.90 + 0.87 + 0.91 + 0.89 + 0.88) / 5

= 0.89
```

So:

```text
CV Score ≈ 0.89
```

---

# 66. `cv=5` Meaning

When you write:

```python
RidgeCV(cv=5)
```

it means the model uses approximately:

```text
5-fold Cross-Validation
```

The exact internal CV behavior can depend on the estimator and scikit-learn configuration.

---

# 67. `scoring` Parameter

You can specify an evaluation metric.

For regression:

```python
scoring="neg_mean_squared_error"
```

Other examples:

```text
r2
neg_mean_absolute_error
neg_root_mean_squared_error
```

For classification:

```python
scoring="accuracy"
```

or:

```text
precision
recall
f1
roc_auc
```

---

# 68. Why Are Regression Errors Negative in Scikit-Learn Scoring?

Scikit-learn generally follows:

```text
higher score = better
```

But:

```text
MSE
```

is better when it is smaller.

Therefore scikit-learn uses:

```text
negative MSE
```

Example:

```text
MSE = 10
score = -10

MSE = 5
score = -5
```

Since:

```text
-5 > -10
```

the smaller MSE is considered better.

---

# 69. `max_iter`

Optimization algorithms may require many iterations.

Example:

```python
LassoCV(
    max_iter=10000
)
```

If you get a convergence warning, increasing:

```text
max_iter
```

may help.

However, convergence warnings can also indicate scaling or other optimization issues.

---

# 70. `tol`

`tol` controls optimization tolerance.

Example:

```python
LassoCV(
    tol=1e-4
)
```

A smaller tolerance generally means stricter convergence criteria and may require more computation.

---

# 71. Common Errors

## Error 1: Convergence Warning

Possible causes:

```text
Features not scaled
max_iter too small
difficult optimization problem
```

Possible solution:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LassoCV(max_iter=10000))
])
```

---

# 72. Error 2: Wrong Target Shape

For:

```text
LassoCV
```

you generally use:

```text
y.shape = (n_samples,)
```

For:

```text
MultiTaskLassoCV
```

you use:

```text
y.shape = (n_samples, n_targets)
```

Example:

```python
print(y.shape)
```

Possible:

```text
LassoCV:
(300,)

MultiTaskLassoCV:
(300, 3)
```

---

# 73. Complete Regression Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(
        alphas=[0.01, 0.1, 1, 10, 100],
        cv=5
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

ridge = model.named_steps["ridge"]

print("Best alpha:", ridge.alpha_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

# 74. Complete LassoCV Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(
        cv=5,
        max_iter=10000
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

lasso = model.named_steps["lasso"]

print("Best alpha:", lasso.alpha_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

# 75. Complete ElasticNetCV Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("elasticnet", ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        cv=5,
        max_iter=10000
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

enet = model.named_steps["elasticnet"]

print("Best alpha:", enet.alpha_)
print("Best l1_ratio:", enet.l1_ratio_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

---

# 76. Complete MultiTaskLassoCV Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import mean_squared_error

X, y = make_regression(
    n_samples=300,
    n_features=20,
    n_targets=3,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", MultiTaskLassoCV(
        cv=5,
        max_iter=10000
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

lasso = model.named_steps["lasso"]

print("Best alpha:", lasso.alpha_)
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

# 77. Complete MultiTaskElasticNetCV Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.metrics import mean_squared_error

X, y = make_regression(
    n_samples=300,
    n_features=20,
    n_targets=3,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MultiTaskElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        cv=5,
        max_iter=10000
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

enet = model.named_steps["model"]

print("Best alpha:", enet.alpha_)
print("Best l1_ratio:", enet.l1_ratio_)
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

# 78. Complete LogisticRegressionCV Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegressionCV(
        Cs=[0.01, 0.1, 1, 10, 100],
        cv=5,
        max_iter=5000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

logistic = model.named_steps["logistic"]

print("Best C:", logistic.C_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

# 79. Cross-Validation Model Selection

A simple decision guide:

```text
Need regression?
       |
       +---- L2 → RidgeCV
       |
       +---- L1 → LassoCV
       |
       +---- L1 + L2 → ElasticNetCV
       |
       +---- Multiple targets?
                  |
                  +---- L1 → MultiTaskLassoCV
                  |
                  +---- L1 + L2 → MultiTaskElasticNetCV

Need classification?
       |
       +---- LogisticRegressionCV
```

---

# 80. When to Choose RidgeCV?

Choose:

```text
RidgeCV
```

when:

* Regression problem
* Many correlated features
* You want L2 regularization
* You want to shrink coefficients
* You don't necessarily want feature elimination

---

# 81. When to Choose LassoCV?

Choose:

```text
LassoCV
```

when:

* Regression problem
* Many features
* Some features may be irrelevant
* Feature selection is desirable
* You want sparse coefficients

---

# 82. When to Choose ElasticNetCV?

Choose:

```text
ElasticNetCV
```

when:

* Regression problem
* Many features
* Features may be correlated
* You want feature selection
* You want both L1 and L2

---

# 83. When to Choose MultiTaskLassoCV?

Choose:

```text
MultiTaskLassoCV
```

when:

* Multiple regression targets
* Targets are related
* Shared feature selection is desirable
* L1-style multi-task regularization is appropriate

---

# 84. When to Choose MultiTaskElasticNetCV?

Choose:

```text
MultiTaskElasticNetCV
```

when:

* Multiple related regression targets
* You want L1 + L2
* You want shared sparsity
* Features are correlated
* Automatic CV selection is desirable

---

# 85. When to Choose LogisticRegressionCV?

Choose:

```text
LogisticRegressionCV
```

when:

* Classification problem
* Linear decision boundary is reasonable
* You want regularization
* You don't want to manually tune `C`
* You need class probabilities

---

# 86. Important Hyperparameter Comparison

| Model                 | Main Hyperparameters                       |
| --------------------- | ------------------------------------------ |
| RidgeCV               | `alphas`, `cv`, `scoring`                  |
| LassoCV               | `alphas`, `cv`, `max_iter`, `tol`          |
| ElasticNetCV          | `alphas`, `l1_ratio`, `cv`, `max_iter`     |
| MultiTaskLassoCV      | `alphas`, `cv`, `max_iter`, `tol`          |
| MultiTaskElasticNetCV | `alphas`, `l1_ratio`, `cv`, `max_iter`     |
| LogisticRegressionCV  | `Cs`, `cv`, `penalty`, `solver`, `scoring` |

---

# 87. Important Attribute Comparison

| Model                 | Best Hyperparameter   |
| --------------------- | --------------------- |
| RidgeCV               | `alpha_`              |
| LassoCV               | `alpha_`              |
| ElasticNetCV          | `alpha_`, `l1_ratio_` |
| MultiTaskLassoCV      | `alpha_`              |
| MultiTaskElasticNetCV | `alpha_`, `l1_ratio_` |
| LogisticRegressionCV  | `C_`                  |

---

# 88. Regularization Cheat Sheet

```text
Ridge
  ↓
L2
  ↓
Σw²

Lasso
  ↓
L1
  ↓
Σ|w|

ElasticNet
  ↓
L1 + L2

MultiTaskLasso
  ↓
Multi-output L1-style group regularization

MultiTaskElasticNet
  ↓
Multi-output L1 + L2

LogisticRegressionCV
  ↓
Classification
  ↓
Regularized logistic loss
```

---

# 89. Alpha Cheat Sheet

For:

```text
RidgeCV
LassoCV
ElasticNetCV
MultiTaskLassoCV
MultiTaskElasticNetCV
```

Generally:

```text
alpha ↑
→ regularization ↑
→ coefficients are more strongly constrained
```

---

# 90. C Cheat Sheet

For:

```text
LogisticRegressionCV
```

Generally:

```text
C ↑
→ regularization ↓

C ↓
→ regularization ↑
```

Remember:

```text
alpha and C behave in opposite directions
```

---

# 91. Cross-Validation vs Train/Test Split

These are not the same.

### Train/Test Split

```text
Training → Model fitting
Testing → Final evaluation
```

### Cross-Validation

```text
Training data
     ↓
Multiple train/validation splits
     ↓
Hyperparameter selection / model evaluation
```

A common workflow is:

```text
Train/Test Split
      ↓
Training Set
      ↓
Cross-Validation
      ↓
Hyperparameter Selection
      ↓
Final Model
      ↓
Test Set Evaluation
```

---

# 92. Cross-Validation vs GridSearchCV

Models like:

```text
RidgeCV
LassoCV
ElasticNetCV
LogisticRegressionCV
```

have built-in CV-based hyperparameter selection.

`GridSearchCV` is a more general tool.

Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

model = Ridge()

params = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

# 93. Built-in CV vs GridSearchCV

### RidgeCV

```python
RidgeCV(
    alphas=[0.1, 1, 10],
    cv=5
)
```

Simple and specialized.

### GridSearchCV

```python
GridSearchCV(
    Ridge(),
    {
        "alpha": [0.1, 1, 10]
    },
    cv=5
)
```

More flexible.

Therefore:

```text
Built-in CV models
→ convenient

GridSearchCV
→ general-purpose hyperparameter search
```

---

# 94. Important Data Leakage Rule

Never tune hyperparameters using the test set.

Wrong:

```text
Test set
   ↓
Choose best alpha
```

Correct:

```text
Training set
   ↓
Cross-Validation
   ↓
Choose alpha
   ↓
Final test
```

---

# 95. Model Complexity

Regularization controls complexity.

Generally:

```text
Weak regularization
→ more complex model
→ potentially lower training error
→ higher overfitting risk

Strong regularization
→ simpler model
→ potentially higher training error
→ potentially lower overfitting
```

Cross-validation helps find a useful balance.

---

# 96. Bias-Variance Perspective

### Too little regularization

```text
Low bias
High variance
```

Potential:

```text
Overfitting
```

### Too much regularization

```text
High bias
Low variance
```

Potential:

```text
Underfitting
```

CV helps select a reasonable regularization strength.

---

# 97. Correlated Features

Suppose:

```text
X1 = Height in cm
X2 = Height in inches
```

They contain nearly the same information.

With highly correlated features:

```text
Ridge
→ generally distributes/shrinks coefficients smoothly

Lasso
→ may select one feature and shrink another to zero

ElasticNet
→ can be a useful compromise
```

---

# 98. Sparse Models

A model is sparse when many coefficients are zero.

Example:

```text
Before:

w = [2.1, 0.7, -1.2, 0.0, 0.0, 3.1]

After feature selection:

Only non-zero coefficients matter.
```

Lasso and ElasticNet can create sparse models.

---

# 99. MultiTask Sparse Learning

MultiTask models extend sparsity to multiple outputs.

Conceptually:

```text
              Target 1   Target 2   Target 3

Feature A        ✓          ✓          ✓
Feature B        ×          ×          ×
Feature C        ✓          ✓          ✓
```

The same features can be encouraged to participate across multiple tasks.

---

# 100. Practical Pipeline Pattern

For regularized linear models:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RidgeCV())
])
```

For classification:

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegressionCV())
])
```

This helps prevent preprocessing leakage during cross-validation.

---

# 101. Model Persistence

After training, save the model.

```python
import joblib

joblib.dump(model, "model.pkl")
```

Load it later:

```python
model = joblib.load("model.pkl")
```

For a Pipeline, the scaler and estimator are saved together.

---

# 102. Production Workflow

A typical production workflow:

```text
Collect Data
    ↓
Clean Data
    ↓
Train/Test Split
    ↓
Build Pipeline
    ↓
Cross-Validation
    ↓
Select Hyperparameters
    ↓
Train Final Model
    ↓
Evaluate Test Set
    ↓
Save Pipeline
    ↓
Deploy
```

---

# 103. Common Mistakes

### Mistake 1

Using test data for hyperparameter tuning.

```text
Wrong ❌
```

### Mistake 2

Scaling before CV manually on the entire dataset.

```text
Wrong ❌
```

Use Pipeline.

### Mistake 3

Forgetting target shape in multi-task learning.

```text
MultiTask:
y → 2D
```

### Mistake 4

Thinking:

```text
alpha ↑
```

means stronger regularization for Logistic Regression.

Wrong.

For Logistic Regression:

```text
C ↓
→ stronger regularization
```

### Mistake 5

Using multi-task models for unrelated targets.

Multi-task learning works best when the tasks have useful relationships.

---

# 104. Interview Questions — Beginner

### Q1. What is Cross-Validation?

Cross-Validation evaluates a model across multiple train/validation splits.

### Q2. What does CV mean in `RidgeCV`?

It means Cross-Validation.

### Q3. What is RidgeCV?

Ridge Regression with automatic alpha selection through Cross-Validation.

### Q4. What is LassoCV?

Lasso Regression with Cross-Validation.

### Q5. What regularization does Ridge use?

L2 regularization.

### Q6. What regularization does Lasso use?

L1 regularization.

### Q7. What is ElasticNet?

A combination of L1 and L2 regularization.

### Q8. What does `alpha` control?

The strength of regularization.

### Q9. What does `l1_ratio` control?

The balance between L1 and L2 regularization in ElasticNet.

### Q10. What is LogisticRegressionCV?

Logistic Regression with Cross-Validation for selecting regularization settings.

---

# 105. Interview Questions — Intermediate

### Q1. Difference between RidgeCV and LassoCV?

```text
RidgeCV → L2
LassoCV → L1
```

### Q2. Why does Lasso perform feature selection?

Because L1 regularization can drive coefficients exactly to zero.

### Q3. Why use ElasticNet instead of Lasso?

ElasticNet combines L1 and L2 and can behave better with correlated features.

### Q4. What is the difference between alpha and C?

```text
alpha ↑ → stronger regularization

C ↑ → weaker regularization
```

### Q5. Why use StandardScaler?

To put features on comparable scales and often improve optimization for regularized linear models.

### Q6. Why use Pipeline?

To ensure preprocessing is performed correctly within each CV training split and reduce data leakage risk.

### Q7. What is MultiTaskLassoCV?

A cross-validated multi-output Lasso-style model that jointly learns related regression tasks.

### Q8. What is MultiTaskElasticNetCV?

A multi-output ElasticNet model using Cross-Validation.

---

# 106. Interview Questions — Advanced

### Q1. Why can Lasso be unstable with highly correlated features?

When multiple features contain similar information, Lasso may arbitrarily favor some and suppress others.

### Q2. Why can ElasticNet help with correlated features?

The L2 component can encourage more stable coefficient behavior while retaining L1 sparsity.

### Q3. Why should scaling be inside the Pipeline?

Because fitting the scaler on all data before CV can leak information from validation folds.

### Q4. Why is a separate test set still necessary if CV is used?

CV is generally used for model selection/tuning; a held-out test set provides an independent final evaluation.

### Q5. Why are negative MSE scores used in scikit-learn?

Because scikit-learn's scoring convention treats larger scores as better.

### Q6. Why are MultiTask models different from simply training separate models?

MultiTask models jointly regularize related targets and can encourage shared feature usage.

### Q7. What is the purpose of `l1_ratio`?

It determines the relative contribution of L1 and L2 penalties in ElasticNet-type models.

---

# 107. Quick Revision Table

| Model                 | Problem                 | Regularization       | CV  | Main Selection        |
| --------------------- | ----------------------- | -------------------- | --- | --------------------- |
| RidgeCV               | Regression              | L2                   | Yes | `alpha_`              |
| LassoCV               | Regression              | L1                   | Yes | `alpha_`              |
| ElasticNetCV          | Regression              | L1 + L2              | Yes | `alpha_`, `l1_ratio_` |
| MultiTaskLassoCV      | Multi-output Regression | Multi-task L1-style  | Yes | `alpha_`              |
| MultiTaskElasticNetCV | Multi-output Regression | Multi-task L1 + L2   | Yes | `alpha_`, `l1_ratio_` |
| LogisticRegressionCV  | Classification          | Regularized logistic | Yes | `C_`                  |

---

# 108. One-Line Definitions

```text
RidgeCV
→ Ridge Regression + Cross-Validation

LassoCV
→ Lasso Regression + Cross-Validation

ElasticNetCV
→ ElasticNet + Cross-Validation

MultiTaskLassoCV
→ Multi-output Lasso-style regression + CV

MultiTaskElasticNetCV
→ Multi-output ElasticNet + CV

LogisticRegressionCV
→ Logistic Regression + Cross-Validation
```

---

# 109. Final Model Selection Cheat Sheet

```text
                 Cross-Validation Models
                          |
          +---------------+---------------+
          |                               |
      Regression                     Classification
          |                               |
    +-----+------+                LogisticRegressionCV
    |     |      |
 Ridge  Lasso  ElasticNet
   |      |        |
RidgeCV LassoCV ElasticNetCV
                    |
              Multiple Targets?
                    |
             +------+------+
             |             |
        MultiTask       MultiTask
         LassoCV      ElasticNetCV
```

---

# 110. Final Summary

Cross-Validation models make hyperparameter selection easier and more systematic.

### RidgeCV

```text
L2 regularization
+
Cross-Validation
+
Automatic alpha selection
```

### LassoCV

```text
L1 regularization
+
Cross-Validation
+
Feature selection
```

### ElasticNetCV

```text
L1 + L2
+
Cross-Validation
+
alpha + l1_ratio selection
```

### MultiTaskLassoCV

```text
Multiple regression targets
+
L1-style shared regularization
+
Cross-Validation
```

### MultiTaskElasticNetCV

```text
Multiple regression targets
+
L1 + L2
+
Cross-Validation
```

### LogisticRegressionCV

```text
Classification
+
Logistic Regression
+
Cross-Validation
+
C selection
```

## Most Important Things to Remember

```text
RidgeCV              → L2
LassoCV              → L1
ElasticNetCV         → L1 + L2
MultiTaskLassoCV     → Multi-output L1-style
MultiTaskElasticNetCV→ Multi-output L1 + L2
LogisticRegressionCV → Classification
```

And:

```text
alpha ↑
→ stronger regularization

C ↑
→ weaker regularization

l1_ratio ↑
→ more L1
→ less L2
```

The most important practical rule:

```text
Train/Test Split
       ↓
Pipeline
       ↓
Cross-Validation
       ↓
Hyperparameter Selection
       ↓
Final Test Evaluation
```

This workflow helps build models that generalize better and reduces the risk of data leakage.
