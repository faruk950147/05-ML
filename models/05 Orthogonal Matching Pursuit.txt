# 5. Orthogonal Matching Pursuit

Scikit-learn provides two important Orthogonal Matching Pursuit models:

1. `OrthogonalMatchingPursuit`
2. `OrthogonalMatchingPursuitCV`

Both are **linear regression models** designed for **sparse solutions**.

---

# 1. What is Orthogonal Matching Pursuit?

**Orthogonal Matching Pursuit (OMP)** is a **greedy algorithm** used for finding a sparse linear representation of data.

In simple words:

> OMP tries to explain the target using only a small number of important features.

Suppose you have 1000 features:

```text
X = [x1, x2, x3, ..., x1000]
```

but only 5 features are actually important.

OMP attempts to find something like:

```text
y ≈ 3.2x17 - 1.5x204 + 2.7x501 + 0.8x700 - 1.1x900
```

Instead of using all 1000 features.

Therefore, OMP is useful when you expect the true model to be **sparse**.

---

# 2. Why is it called "Matching Pursuit"?

The algorithm repeatedly looks for the feature that has the strongest relationship with the current residual.

The basic idea is:

```text
Start
  ↓
Calculate residual
  ↓
Find feature most correlated with residual
  ↓
Add that feature
  ↓
Refit coefficients
  ↓
Calculate new residual
  ↓
Repeat
```

The algorithm "pursues" the features that best explain the remaining error.

---

# 3. What does "Orthogonal" mean?

After selecting features, OMP performs a least-squares fit using the selected features.

This means the residual becomes orthogonal to the selected feature space.

Conceptually:

```text
Selected features
       ↓
Least-squares fitting
       ↓
Residual
       ↓
Residual is orthogonal to selected feature space
```

This is what distinguishes **Orthogonal Matching Pursuit** from simpler greedy matching-pursuit approaches.

---

# 4. Sparse Regression

Before understanding OMP, you should understand **sparsity**.

Suppose:

```text
β = [0, 0, 2.5, 0, 0, -1.7, 0, 0, 0.8]
```

Most coefficients are zero.

This is a **sparse coefficient vector**.

Compare:

### Dense model

```text
β = [1.2, -0.7, 2.3, 0.4, -1.1, 0.8]
```

Almost every feature has a non-zero coefficient.

### Sparse model

```text
β = [0, 0, 2.3, 0, 0, 0.8]
```

Only a few features are used.

OMP attempts to produce this type of sparse solution.

---

# 5. OrthogonalMatchingPursuit

Scikit-learn provides:

```python
from sklearn.linear_model import OrthogonalMatchingPursuit
```

Basic syntax:

```python
model = OrthogonalMatchingPursuit()
```

Then:

```python
model.fit(X_train, y_train)
```

Prediction:

```python
y_pred = model.predict(X_test)
```

---

# 6. Basic Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import OrthogonalMatchingPursuit

X, y = make_regression(
    n_samples=100,
    n_features=20,
    n_informative=5,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", y_pred[:5])
```

Here:

```python
n_nonzero_coefs=5
```

means we want approximately 5 non-zero coefficients.

---

# 7. Understanding n_nonzero_coefs

One of the most important parameters is:

```python
n_nonzero_coefs
```

It controls the maximum number of non-zero coefficients.

Example:

```python
model = OrthogonalMatchingPursuit(
    n_nonzero_coefs=3
)
```

The model attempts to use at most 3 active features.

For example:

```text
Feature 0 → 0
Feature 1 → 0
Feature 2 → 4.2
Feature 3 → 0
Feature 4 → -2.1
Feature 5 → 0
Feature 6 → 1.7
```

Only:

```text
Feature 2
Feature 4
Feature 6
```

are active.

---

# 8. Why use OMP?

OMP is useful when:

* You expect a sparse model.
* You have many features.
* Only a small number of features are expected to matter.
* Interpretability is important.
* Feature selection is useful.
* You want a greedy sparse approximation algorithm.

Typical applications include:

* Signal processing
* Compressed sensing
* Sparse coding
* Feature selection
* Dictionary learning
* High-dimensional regression
* Scientific modeling

---

# 9. OMP Algorithm — Step by Step

Suppose:

```text
X = [x1, x2, x3, x4]
```

and:

```text
y = target
```

Initially:

```text
Selected features = []
```

and:

```text
residual = y
```

## Step 1: Find the best feature

Calculate correlation between every feature and residual.

Suppose:

```text
corr(x1, residual) = 0.2
corr(x2, residual) = 0.8
corr(x3, residual) = 0.1
corr(x4, residual) = 0.4
```

OMP selects:

```text
x2
```

because it has the largest absolute correlation.

---

## Step 2: Fit selected features

Now:

```text
Selected = [x2]
```

Fit:

```text
y ≈ β2x2
```

Calculate new residual:

```text
r = y - β2x2
```

---

## Step 3: Find next feature

Now calculate correlations with the new residual.

Suppose:

```text
x1 → 0.3
x3 → 0.7
x4 → 0.2
```

Select:

```text
x3
```

Now:

```text
Selected = [x2, x3]
```

---

## Step 4: Refit

Fit:

```text
y ≈ β2x2 + β3x3
```

Then calculate a new residual.

---

## Step 5: Continue

The process continues until:

```text
number of selected features
```

reaches the specified value.

For example:

```python
n_nonzero_coefs=5
```

Then approximately:

```text
5 features → selected
remaining features → coefficient 0
```

---

# 10. Important Parameters of OrthogonalMatchingPursuit

Main parameters include:

```python
OrthogonalMatchingPursuit(
    n_nonzero_coefs=None,
    tol=None,
    fit_intercept=True,
    precompute="auto"
)
```

---

## 10.1 n_nonzero_coefs

Controls the number of non-zero coefficients.

Example:

```python
model = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)
```

Smaller value:

```text
More sparse
Less complex
Potentially higher bias
```

Larger value:

```text
Less sparse
More complex
Potentially lower training error
```

---

# 11. tol

`tol` is a tolerance used as a stopping criterion.

Example:

```python
model = OrthogonalMatchingPursuit(
    tol=1e-5
)
```

Conceptually:

```text
If residual becomes sufficiently small
        ↓
Stop selecting features
```

When `tol` is specified, it can determine when the algorithm should stop based on the residual rather than simply selecting a fixed number of coefficients.

---

# 12. n_nonzero_coefs vs tol

These parameters represent two different ways of controlling the model.

### n_nonzero_coefs

```text
"Select approximately this many features."
```

### tol

```text
"Stop when the residual/error becomes sufficiently small."
```

Conceptually:

```text
n_nonzero_coefs → complexity-based stopping
tol             → error-based stopping
```

---

# 13. fit_intercept

Default:

```python
fit_intercept=True
```

If:

```python
fit_intercept=True
```

the model estimates:

```text
intercept
```

If:

```python
fit_intercept=False
```

the model assumes:

```text
intercept = 0
```

Example:

```python
model = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5,
    fit_intercept=False
)
```

---

# 14. precompute

OMP can use precomputed information to improve computational efficiency.

Example:

```python
model = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5,
    precompute=True
)
```

Depending on the data and scikit-learn version, `precompute` may be automatically selected with:

```python
precompute="auto"
```

Usually, beginners do not need to change this parameter.

---

# 15. Understanding coef_

After training:

```python
model.coef_
```

Example:

```python
print(model.coef_)
```

Possible output:

```text
[0.0, 0.0, 2.5, 0.0, -1.7, 0.0, 0.0, 3.2]
```

This immediately tells us which features are active.

Non-zero:

```text
Feature 2
Feature 4
Feature 7
```

Zero:

```text
Feature 0
Feature 1
Feature 3
Feature 5
Feature 6
```

This is why OMP can be highly interpretable.

---

# 16. OMP as Feature Selection

OMP performs feature selection as part of its algorithm.

Suppose:

```text
100 features
```

but:

```text
n_nonzero_coefs=5
```

The final model may use only:

```text
5 features
```

Therefore:

```text
OMP
↓
Select important features
↓
Fit coefficients
↓
Produce sparse regression model
```

---

# 17. Measuring Sparsity

You can count the number of non-zero coefficients:

```python
import numpy as np

non_zero = np.sum(model.coef_ != 0)

print("Non-zero coefficients:", non_zero)
```

You can also inspect:

```python
print(model.coef_)
```

---

# 18. Complete OMP Example with Evaluation

```python
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(
    n_samples=300,
    n_features=30,
    n_informative=5,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)

print("Coefficients:")
print(model.coef_)

print(
    "Number of non-zero coefficients:",
    np.sum(model.coef_ != 0)
)
```

---

# 19. OrthogonalMatchingPursuitCV

Scikit-learn also provides:

```python
from sklearn.linear_model import OrthogonalMatchingPursuitCV
```

The important difference is:

> `OrthogonalMatchingPursuitCV` automatically chooses the number of non-zero coefficients using cross-validation.

Instead of manually deciding:

```python
n_nonzero_coefs=5
```

we can allow cross-validation to find a suitable value.

---

# 20. Basic Syntax

```python
from sklearn.linear_model import OrthogonalMatchingPursuitCV

model = OrthogonalMatchingPursuitCV()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 21. Why use OrthogonalMatchingPursuitCV?

Suppose you don't know whether the model should use:

```text
3 features
5 features
8 features
10 features
```

Choosing manually can be difficult.

OMP-CV evaluates different sparsity levels using cross-validation.

Conceptually:

```text
Try 1 feature
      ↓
Cross-validation

Try 2 features
      ↓
Cross-validation

Try 3 features
      ↓
Cross-validation

...
      ↓
Choose best number
```

Therefore:

```text
OrthogonalMatchingPursuit
        ↓
You control sparsity

OrthogonalMatchingPursuitCV
        ↓
Cross-validation controls sparsity
```

---

# 22. Basic OMP-CV Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.metrics import mean_squared_error, r2_score

X, y = make_regression(
    n_samples=300,
    n_features=30,
    n_informative=5,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = OrthogonalMatchingPursuitCV(
    cv=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

print("Selected coefficients:")
print(model.coef_)

print("Selected number of non-zero coefficients:",
      sum(model.coef_ != 0))
```

---

# 23. cv Parameter

`cv` controls the cross-validation strategy.

Example:

```python
model = OrthogonalMatchingPursuitCV(
    cv=5
)
```

This means:

```text
5-fold cross-validation
```

Conceptually:

```text
Dataset
   ↓
Split into 5 folds

Fold 1 → validation
Fold 2 → validation
Fold 3 → validation
Fold 4 → validation
Fold 5 → validation
```

Each candidate sparsity level is evaluated.

---

# 24. max_iter

`OrthogonalMatchingPursuitCV` can also use:

```python
max_iter
```

to control the maximum number of iterations considered.

Example:

```python
model = OrthogonalMatchingPursuitCV(
    cv=5,
    max_iter=20
)
```

This can limit the maximum number of non-zero coefficients considered.

---

# 25. n_nonzero_coefs in OMP-CV

The important conceptual distinction:

### OMP

```python
OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)
```

You specify the sparsity level.

### OMP-CV

```python
OrthogonalMatchingPursuitCV(
    cv=5
)
```

Cross-validation selects an appropriate sparsity level.

---

# 26. OMP vs OMP-CV

| Feature                  | OrthogonalMatchingPursuit | OrthogonalMatchingPursuitCV   |
| ------------------------ | ------------------------- | ----------------------------- |
| Algorithm                | OMP                       | OMP + CV                      |
| Sparse model             | Yes                       | Yes                           |
| Feature selection        | Yes                       | Yes                           |
| Number of coefficients   | User-controlled           | CV-selected                   |
| Cross-validation         | No                        | Yes                           |
| Computation              | Faster                    | Usually slower                |
| Hyperparameter selection | Manual                    | Automatic                     |
| Use when                 | You know desired sparsity | You want data-driven sparsity |

---

# 27. OMP vs LinearRegression

`LinearRegression` normally uses all available features.

OMP tries to use only a limited number.

### LinearRegression

```text
X
↓
Use all features
↓
Fit coefficients
```

### OMP

```text
X
↓
Select useful features
↓
Fit coefficients
↓
Sparse model
```

Example:

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
```

versus:

```python
from sklearn.linear_model import OrthogonalMatchingPursuit

omp = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)

omp.fit(X_train, y_train)
```

---

# 28. OMP vs Lasso

Both can create sparse models, but their approaches are different.

## Lasso

Lasso uses an L1 regularization penalty:

```text
Loss + α × L1 penalty
```

This encourages coefficients to become zero.

## OMP

OMP uses a greedy feature-selection process.

```text
Find best feature
↓
Add it
↓
Refit
↓
Find next feature
↓
Repeat
```

Comparison:

| Feature                  | OMP                            | Lasso                         |
| ------------------------ | ------------------------------ | ----------------------------- |
| Sparse                   | Yes                            | Yes                           |
| Main idea                | Greedy selection               | L1 regularization             |
| Feature selection        | Explicit                       | Through coefficient shrinkage |
| Hyperparameter           | Number of features / tolerance | alpha                         |
| Cross-validation version | OMP-CV                         | LassoCV                       |
| Optimization             | Greedy                         | Convex optimization           |
| Useful for               | Sparse representation          | Regularized regression        |

---

# 29. OMP vs LassoCV

### OMP

```python
OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)
```

You specify sparsity.

### OMP-CV

```python
OrthogonalMatchingPursuitCV(
    cv=5
)
```

CV chooses sparsity.

### LassoCV

```python
from sklearn.linear_model import LassoCV

model = LassoCV(cv=5)
```

LassoCV selects:

```text
alpha
```

using cross-validation.

Therefore:

```text
OMP-CV → chooses sparsity level

LassoCV → chooses regularization strength
```

---

# 30. OMP vs Ridge

Ridge regression uses L2 regularization:

```text
Loss + α × ||β||²
```

Ridge generally shrinks coefficients toward zero but does not normally make them exactly zero.

OMP aims for:

```text
many coefficients = 0
```

Therefore:

```text
Ridge → shrinkage

OMP → sparse feature selection
```

---

# 31. OMP vs ElasticNet

ElasticNet combines:

```text
L1 + L2
```

OMP uses:

```text
greedy feature selection + least squares
```

ElasticNet can be useful when:

* Features are correlated.
* You want regularization.
* You want sparse coefficients.

OMP can be useful when:

* You strongly expect sparsity.
* You want explicit feature selection.
* You want a greedy sparse approximation.

---

# 32. Mathematical Intuition

The standard linear regression model is:

$$
y = X\beta + \epsilon
$$

where:

* \(X\) = feature matrix
* \(y\) = target
* \(\beta\) = coefficient vector
* \(\epsilon\) = error

OMP searches for a sparse coefficient vector:

$$
\beta
$$

such that:

$$
y \approx X\beta
$$

while using only a small number of non-zero coefficients.

---

# 33. Residual

The residual is:

$$
r = y - X\beta
$$

Initially, there are no selected features, so conceptually:

$$
r_0 = y
$$

OMP searches for the feature that has the largest correlation with the current residual.

Conceptually:

$$
j^* =
\arg\max_j
|X_j^T r|
$$

where:

* \(X_j\) = candidate feature
* \(r\) = current residual

The selected feature is added to the active set.

---

# 34. Active Set

OMP maintains a set of selected features.

For example:

```text
Initially:

Active set = {}

After iteration 1:

Active set = {x5}

After iteration 2:

Active set = {x5, x12}

After iteration 3:

Active set = {x5, x12, x20}
```

Then least squares is solved using only these selected features.

---

# 35. Why Orthogonal?

Suppose selected features are:

```text
X_selected
```

OMP solves:

$$
\hat{\beta}
=
\arg\min_\beta
\|y-X_{selected}\beta\|_2^2
$$

After the least-squares solution, the residual is orthogonal to the selected feature space.

Conceptually:

$$
X_{selected}^T r = 0
$$

This helps OMP avoid simply repeating the same information already explained by selected features.

---

# 36. OMP Algorithm

A simplified algorithm:

```text
Input:
    X
    y
    desired sparsity k

Initialize:
    residual = y
    selected_features = []

Repeat k times:

    1. Calculate correlation between
       every unused feature and residual.

    2. Select feature with maximum
       absolute correlation.

    3. Add feature to selected set.

    4. Solve least squares using
       selected features.

    5. Update residual.

Return:
    sparse coefficient vector
```

---

# 37. Why OMP Can Be Fast

OMP does not search every possible combination of features.

If you have:

```text
100 features
```

there are an enormous number of possible feature subsets.

OMP uses a greedy strategy:

```text
Choose best feature now
↓
Choose next best feature
↓
Continue
```

This makes the problem much more manageable.

However, computational cost can still become significant with very large datasets or feature spaces.

---

# 38. Limitations of OMP

OMP is not always the best model.

Important limitations:

### 1. Greedy algorithm

OMP makes decisions step by step.

A feature selected early may not be globally optimal.

### 2. Correlated features

If several features are highly correlated, OMP may select one feature somewhat arbitrarily.

### 3. Noisy data

Strong noise can affect correlation-based feature selection.

### 4. Computational cost

Repeated least-squares fitting can become expensive for large problems.

### 5. Sparsity assumption

OMP works best when the underlying relationship is reasonably sparse.

---

# 39. OMP and Correlated Features

Suppose:

```text
x1 ≈ x2 ≈ x3
```

All three contain almost the same information.

OMP might select:

```text
x2
```

while another run or another dataset might favor:

```text
x1
```

This means feature selection can be unstable when features are strongly correlated.

---

# 40. OMP and Feature Scaling

Feature scaling can matter for OMP because feature selection is based on correlations/inner products.

If features have very different scales:

```text
Feature A: 0–1
Feature B: 0–100000
```

the larger-scale feature can dominate calculations.

A common approach is:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuit

model = make_pipeline(
    StandardScaler(),
    OrthogonalMatchingPursuit(n_nonzero_coefs=5)
)
```

This standardizes features before applying OMP.

---

# 41. OMP Pipeline Example

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuit

model = make_pipeline(
    StandardScaler(),
    OrthogonalMatchingPursuit(
        n_nonzero_coefs=5
    )
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

This is especially useful when the features have different units or scales.

---

# 42. OMP-CV Pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuitCV

model = make_pipeline(
    StandardScaler(),
    OrthogonalMatchingPursuitCV(cv=5)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

The scaling happens inside the training process, which helps avoid data leakage.

---

# 43. Data Leakage Warning

Avoid:

```python
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train, X_test = train_test_split(
    X_scaled,
    test_size=0.2
)
```

A safer approach is:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = make_pipeline(
    StandardScaler(),
    OrthogonalMatchingPursuitCV(cv=5)
)

model.fit(X_train, y_train)
```

The pipeline learns scaling parameters only from the training data.

---

# 44. Sparse Signal Example

Imagine a signal is generated from 100 possible basis functions:

```text
100 candidate signals
```

but only:

```text
4 signals
```

are actually present.

Then:

```text
y =
3.5x10
- 2.1x37
+ 1.8x62
+ 4.2x91
```

OMP is a natural candidate because the true representation is sparse.

---

# 45. Compressed Sensing

OMP is commonly associated with **compressed sensing**.

The basic idea is:

```text
Large signal
     ↓
Only a few important components
     ↓
Sparse representation
     ↓
Recover using limited measurements
```

OMP can be used as one of the algorithms for recovering sparse signals.

---

# 46. Sparse Coding

Suppose you have a dictionary:

```text
D = [d1, d2, d3, ..., d1000]
```

and want to represent a signal:

```text
y
```

using only a few dictionary atoms:

```text
y ≈ a1d10 + a2d200 + a3d700
```

OMP can select the relevant dictionary atoms.

---

# 47. Feature Selection Example

Suppose:

```text
1000 features
```

and you believe only:

```text
10 features
```

are useful.

You can try:

```python
model = OrthogonalMatchingPursuit(
    n_nonzero_coefs=10
)
```

After training:

```python
selected_features = np.flatnonzero(model.coef_)

print(selected_features)
```

This returns the indices of selected features.

---

# 48. OMP-CV Automatic Feature Selection

With:

```python
model = OrthogonalMatchingPursuitCV(cv=5)
```

the model can determine an appropriate sparsity level through cross-validation.

Then:

```python
selected_features = np.flatnonzero(model.coef_)

print(selected_features)
```

You can inspect which features were selected.

---

# 49. Multi-Output Regression

OMP can also work with multi-target regression in scikit-learn.

For example:

```text
X → [features]

y → [
    target1,
    target2
]
```

However, it is important to understand that this is not the same concept as specialized multi-task regularization models such as:

```text
MultiTaskLasso
MultiTaskElasticNet
```

OMP's behavior and API should be checked for the specific scikit-learn version and use case when working with multi-output targets.

---

# 50. OMP Model Workflow

A practical workflow:

```text
1. Load dataset
       ↓
2. Split train/test
       ↓
3. Scale features if appropriate
       ↓
4. Choose OMP or OMP-CV
       ↓
5. Train
       ↓
6. Predict
       ↓
7. Evaluate
       ↓
8. Inspect non-zero coefficients
       ↓
9. Analyze selected features
```

---

# 51. When Should You Use OMP?

Use OMP when:

```text
✓ You expect a sparse solution
✓ Only a few features should matter
✓ Feature selection is important
✓ Interpretability is important
✓ Sparse representation is required
✓ You work with signal-processing problems
✓ You want a greedy sparse approximation method
```

---

# 52. When Should You Avoid OMP?

Consider other models when:

```text
✗ Most features are expected to matter
✗ Features are highly correlated
✗ You need strong regularization stability
✗ The data is extremely noisy
✗ You don't have a reasonable sparsity assumption
```

In such cases, consider:

```text
Ridge
Lasso
ElasticNet
```

or other appropriate models.

---

# 53. OMP vs Major Linear Models

| Model            | Main idea                |    Sparse? | Main control          |
| ---------------- | ------------------------ | ---------: | --------------------- |
| LinearRegression | Ordinary least squares   |         No | None                  |
| Ridge            | L2 regularization        | Usually no | alpha                 |
| Lasso            | L1 regularization        |        Yes | alpha                 |
| ElasticNet       | L1 + L2                  |        Yes | alpha, l1_ratio       |
| OMP              | Greedy feature selection |        Yes | n_nonzero_coefs / tol |
| OMP-CV           | OMP + cross-validation   |        Yes | CV-selected sparsity  |

---

# 54. OMP vs MultiTaskLasso

These models both involve sparsity, but they solve different problems.

### OMP

Focus:

```text
Greedy sparse feature selection
```

### MultiTaskLasso

Focus:

```text
Multiple related regression tasks
+
shared feature selection
```

If you have:

```text
X → multiple targets
```

and believe the same features are useful across tasks, MultiTaskLasso can be appropriate.

---

# 55. OMP vs OrthogonalMatchingPursuitCV

Remember this simple rule:

```text
OMP
    ↓
Manual sparsity

OMP-CV
    ↓
Automatic sparsity selection
```

Example:

```python
omp = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)
```

versus:

```python
omp_cv = OrthogonalMatchingPursuitCV(
    cv=5
)
```

---

# 56. Practical Example: Compare Different Sparsity Levels

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import r2_score

X, y = make_regression(
    n_samples=500,
    n_features=50,
    n_informative=8,
    noise=15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

for k in [3, 5, 8, 10, 15]:

    model = OrthogonalMatchingPursuit(
        n_nonzero_coefs=k
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(
        "k =", k,
        "R² =", r2_score(y_test, y_pred)
    )
```

This lets you observe how changing the sparsity level affects performance.

---

# 57. Choosing n_nonzero_coefs

There is no universal best value.

Possible strategies:

### Strategy 1 — Domain knowledge

If you know approximately 5 features should matter:

```python
n_nonzero_coefs=5
```

### Strategy 2 — Experimentation

Try:

```text
3
5
8
10
15
```

and compare validation performance.

### Strategy 3 — OMP-CV

Let cross-validation choose the sparsity level:

```python
OrthogonalMatchingPursuitCV(cv=5)
```

---

# 58. Model Evaluation

For regression, common metrics include:

### Mean Squared Error

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
$$

Python:

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

Lower is better.

---

### Mean Absolute Error

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat y_i|
$$

Python:

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

Lower is better.

---

### R²

$$
R^2 =
1 -
\frac{\sum(y_i-\hat y_i)^2}
{\sum(y_i-\bar y)^2}
$$

Python:

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

Higher is generally better.

---

# 59. Complete Practical Example

```python
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Create sparse-like dataset
X, y = make_regression(
    n_samples=500,
    n_features=50,
    n_informative=8,
    noise=10,
    random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# OMP with cross-validation
model = make_pipeline(
    StandardScaler(),
    OrthogonalMatchingPursuitCV(cv=5)
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)
```

---

# 60. Important Attribute: coef_

After fitting an OMP model:

```python
model.coef_
```

For a direct OMP model:

```python
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
omp.fit(X_train, y_train)

print(omp.coef_)
```

For an OMP-CV model:

```python
omp_cv = OrthogonalMatchingPursuitCV(cv=5)
omp_cv.fit(X_train, y_train)

print(omp_cv.coef_)
```

The coefficient vector shows which features are active.

---

# 61. Important Attribute: intercept_

```python
model.intercept_
```

This represents the estimated intercept.

Prediction follows:

$$
\hat y = X\beta + b
$$

where:

* \(X\) = features
* \(\beta\) = coefficients
* \(b\) = intercept

---

# 62. Feature Selection with OMP

Example:

```python
omp = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)

omp.fit(X, y)

selected_features = np.flatnonzero(
    omp.coef_
)

print("Selected features:")
print(selected_features)
```

This is one of the most useful practical aspects of OMP.

---

# 63. Important Concept: Sparse Does Not Automatically Mean Better

A model with fewer features is not automatically more accurate.

For example:

```text
3 features → R² = 0.72
8 features → R² = 0.91
20 features → R² = 0.92
```

You need to balance:

```text
Accuracy
+
Sparsity
+
Interpretability
```

OMP is especially attractive when a compact model is valuable.

---

# 64. OMP and Overfitting

Increasing the number of selected features can make the model more flexible.

Conceptually:

```text
Too few features
      ↓
Underfitting

Good number of features
      ↓
Good generalization

Too many features
      ↓
Potential overfitting
```

OMP-CV can help choose a useful sparsity level based on validation performance.

---

# 65. OMP-CV Workflow

```text
Training data
      ↓
Choose candidate sparsity levels
      ↓
Cross-validation
      ↓
Evaluate each level
      ↓
Select suitable sparsity
      ↓
Fit final OMP model
      ↓
Predict
```

This is why OMP-CV is convenient when you do not know the correct number of features in advance.

---

# 66. Advantages of OrthogonalMatchingPursuit

### 1. Sparse solution

Produces a model using relatively few features.

### 2. Feature selection

Selected features are easy to inspect.

### 3. Interpretable

A sparse model can be easier to understand.

### 4. Useful for sparse signals

Particularly suitable for signal-processing and sparse-representation problems.

### 5. Greedy approach

Can be attractive when a full combinatorial search would be too expensive.

---

# 67. Advantages of OrthogonalMatchingPursuitCV

Everything above, plus:

### Automatic sparsity selection

Instead of manually deciding:

```text
k = 3?
k = 5?
k = 10?
```

cross-validation helps choose the sparsity level.

---

# 68. Disadvantages

### OMP

```text
- Greedy
- Can be sensitive to correlated features
- Sensitive to noise
- Requires a reasonable sparsity assumption
- Choosing sparsity manually can be difficult
```

### OMP-CV

```text
- More computationally expensive
- Cross-validation increases training time
- Still inherits OMP's limitations
```

---

# 69. Interview Questions

## Beginner

### Q1. What is Orthogonal Matching Pursuit?

OMP is a greedy algorithm for finding sparse linear representations by iteratively selecting features that are most correlated with the current residual and refitting using least squares.

---

### Q2. Is OMP a regression model?

Yes. Scikit-learn provides OMP as a linear regression estimator that produces sparse coefficients.

---

### Q3. What is sparsity?

Sparsity means that most model coefficients are zero.

Example:

```text
[0, 0, 2.4, 0, -1.7, 0]
```

---

### Q4. What does n_nonzero_coefs mean?

It controls the desired maximum number of non-zero coefficients selected by OMP.

---

### Q5. What is OrthogonalMatchingPursuitCV?

It is an OMP estimator that uses cross-validation to choose an appropriate sparsity level.

---

# 70. Intermediate Interview Questions

### Q6. How does OMP select features?

It selects the unused feature with the largest absolute correlation with the current residual.

---

### Q7. Why is OMP called orthogonal?

Because after selecting features, it performs least-squares fitting, making the residual orthogonal to the selected feature space.

---

### Q8. OMP vs Lasso?

OMP uses greedy feature selection, while Lasso uses L1 regularization to encourage coefficients to become zero.

---

### Q9. OMP vs Ridge?

Ridge mainly shrinks coefficients using L2 regularization, while OMP explicitly constructs a sparse model.

---

### Q10. What is OMP-CV used for?

It is useful when you don't know the appropriate number of non-zero coefficients and want cross-validation to select it.

---

# 71. Advanced Interview Questions

### Q11. What is the OMP selection criterion?

A simplified form is:

$$
j^* =
\arg\max_j |X_j^T r|
$$

where \(r\) is the current residual.

---

### Q12. What is the residual?

$$
r = y-X\beta
$$

It represents the part of the target that has not yet been explained by the current model.

---

### Q13. What is an active set?

The active set contains the features selected by OMP so far.

Example:

```text
Active set = {x3, x7, x15}
```

---

### Q14. Why can correlated features be problematic for OMP?

When several features contain similar information, their correlations with the residual can be similar. OMP may select one feature while another correlated feature could have been chosen.

---

### Q15. When would you choose OMP-CV instead of OMP?

Use OMP-CV when you don't know the correct sparsity level and want a data-driven selection using cross-validation.

---

# 72. Important Code Cheat Sheet

## OMP

```python
from sklearn.linear_model import OrthogonalMatchingPursuit

model = OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## OMP-CV

```python
from sklearn.linear_model import OrthogonalMatchingPursuitCV

model = OrthogonalMatchingPursuitCV(
    cv=5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## Check coefficients

```python
print(model.coef_)
```

---

## Find selected features

```python
import numpy as np

selected_features = np.flatnonzero(
    model.coef_
)

print(selected_features)
```

---

## Count selected features

```python
count = np.sum(model.coef_ != 0)

print(count)
```

---

# 73. Quick Revision

```text
Orthogonal Matching Pursuit
        ↓
Greedy sparse regression
        ↓
Select feature most correlated with residual
        ↓
Refit using selected features
        ↓
Update residual
        ↓
Repeat
```

### OrthogonalMatchingPursuit

```text
Manual sparsity
```

Example:

```python
OrthogonalMatchingPursuit(
    n_nonzero_coefs=5
)
```

### OrthogonalMatchingPursuitCV

```text
Automatic sparsity selection using CV
```

Example:

```python
OrthogonalMatchingPursuitCV(
    cv=5
)
```

---

# 74. One-Line Cheat Sheet

```text
OrthogonalMatchingPursuit
= Greedy sparse linear regression.

OrthogonalMatchingPursuitCV
= OMP + Cross-Validation for selecting sparsity.

n_nonzero_coefs
= Controls how many coefficients can be non-zero.

tol
= Residual-based stopping criterion.

coef_
= Learned feature coefficients.

intercept_
= Learned intercept.

OMP
= Manual sparsity control.

OMP-CV
= Automatic sparsity selection.

OMP vs Lasso
= Greedy selection vs L1 regularization.

OMP vs Ridge
= Sparse selection vs coefficient shrinkage.
```

---

# 75. Final Summary

Orthogonal Matching Pursuit is a **sparse linear regression technique** based on a greedy feature-selection strategy.

The core process is:

```text
Calculate residual
      ↓
Find feature most correlated with residual
      ↓
Select feature
      ↓
Perform least-squares fitting
      ↓
Update residual
      ↓
Repeat
```

The two main scikit-learn models are:

```text
1. OrthogonalMatchingPursuit
2. OrthogonalMatchingPursuitCV
```

The key difference is:

```text
OrthogonalMatchingPursuit
        ↓
You specify sparsity

OrthogonalMatchingPursuitCV
        ↓
Cross-validation selects sparsity
```

### Remember

```text
OMP
→ Greedy
→ Sparse
→ Feature selection
→ Residual-based selection
→ Least-squares refitting

OMP-CV
→ OMP
→ Cross-validation
→ Automatic sparsity selection
```

If you remember only one concept:

> **OMP repeatedly selects the feature that best explains the current residual and builds a sparse regression model.**
