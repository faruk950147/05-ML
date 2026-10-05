# Multi-Task Models — Full Notes

## Beginner → Intermediate → Advanced

### Models Covered

```text
MultiTaskLasso
MultiTaskElasticNet
```

These models are designed for **multi-output / multi-task regression** problems.

---

# 1. What is Multi-Task Learning?

**Multi-Task Learning (MTL)** is a machine learning approach where a single model learns multiple related prediction tasks simultaneously.

Instead of building completely separate models:

```text
X → Model 1 → y₁
X → Model 2 → y₂
X → Model 3 → y₃
```

Multi-task learning tries to learn them jointly:

```text
             ┌──→ y₁
X → Model ───┼──→ y₂
             └──→ y₃
```

The main idea is:

> Related tasks can share useful information.

---

# 2. Single-Task vs Multi-Task

## Single-Task Learning

One input produces one target.

```text
X → y
```

Example:

```text
House features → House price
```

---

## Multi-Task Learning

One input produces multiple related targets.

```text
X → y₁
  → y₂
  → y₃
```

Example:

```text
House features
     ↓
 ┌───────────────┐
 │ Multi-Task    │
 │ Regression    │
 └───────────────┘
     ↓
 ├── House price
 ├── Monthly rent
 └── Property tax
```

---

# 3. What is Multi-Output Regression?

Multi-output regression means that the target contains multiple continuous values for each sample.

For example:

```text
X.shape
= (1000, 20)
```

means:

```text
1000 samples
20 features
```

And:

```text
y.shape
= (1000, 3)
```

means:

```text
1000 samples
3 target variables
```

Example:

```text
y₁ = house price
y₂ = annual rent
y₃ = property value
```

---

# 4. Why Use Multi-Task Models?

Suppose three tasks are related:

```text
Task 1 → Predict blood pressure
Task 2 → Predict cholesterol
Task 3 → Predict heart rate
```

They may share useful input information.

Instead of learning them independently:

```text
Model 1 → Task 1
Model 2 → Task 2
Model 3 → Task 3
```

a multi-task model learns them jointly.

Potential benefits:

* Shared information between tasks
* Better generalization when tasks are related
* Shared feature selection
* Reduced need for separate models
* More compact modeling

---

# 5. Main Multi-Task Models

Scikit-learn provides:

```python
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskElasticNet
```

The two models are:

```text
MultiTaskLasso
MultiTaskElasticNet
```

---

# 6. MultiTaskLasso

## 6.1 What is MultiTaskLasso?

`MultiTaskLasso` is a linear regression model designed for multiple regression targets.

It uses a **joint L1-style regularization** across tasks.

```python
from sklearn.linear_model import MultiTaskLasso
```

Basic syntax:

```python
model = MultiTaskLasso(alpha=1.0)
```

---

# 7. MultiTaskLasso Concept

Suppose we have:

```text
X
 ↓
 ┌───────────────┐
 │ MultiTaskLasso│
 └───────────────┘
 ↓
 ├── Target 1
 ├── Target 2
 └── Target 3
```

The model learns coefficient matrix:

```text
W =
[
 w11  w12  w13
 w21  w22  w23
 w31  w32  w33
 ...
]
```

Rows correspond to features.

Columns correspond to tasks.

---

# 8. MultiTaskLasso Mathematical Form

Let:

```text
Y = XW + E
```

where:

```text
X = input feature matrix
Y = multiple target matrix
W = coefficient matrix
E = error
```

MultiTaskLasso minimizes a loss with a mixed L1-style penalty.

Conceptually:

```text
Loss
+
α × Σᵢ ||Wᵢ||₂
```

where `Wᵢ` represents the coefficients of feature `i` across all tasks.

The important idea is:

```text
Feature i
   ↓
Coefficients across all tasks
   ↓
Group regularization
```

---

# 9. Why is MultiTaskLasso Different from Lasso?

Ordinary Lasso treats targets independently.

```text
Lasso:

X → y₁
X → y₂
X → y₃
```

If trained separately, each model can select different features.

MultiTaskLasso instead encourages a shared feature-selection structure:

```text
MultiTaskLasso:

Feature 1 → Task 1 ✓
           Task 2 ✓
           Task 3 ✓

Feature 2 → Task 1 ×
           Task 2 ×
           Task 3 ×
```

So the model encourages features to be selected or removed across tasks together.

---

# 10. MultiTaskLasso Example

Let's create a multi-output regression dataset.

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLasso
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

model = MultiTaskLasso(
    alpha=0.1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```

---

# 11. Checking MultiTaskLasso Coefficients

```python
print(model.coef_)
```

The shape is:

```text
(n_targets, n_features)
```

For example:

```text
(3, 20)
```

means:

```text
3 targets
20 features
```

The coefficient matrix conceptually looks like:

```text
             Feature 1 Feature 2 Feature 3 ...

Target 1       w11       w12       w13
Target 2       w21       w22       w23
Target 3       w31       w32       w33
```

---

# 12. MultiTaskLasso Parameters

Important parameters include:

```text
alpha
fit_intercept
max_iter
tol
selection
random_state
```

Example:

```python
model = MultiTaskLasso(
    alpha=0.1,
    max_iter=10000,
    tol=1e-4,
    random_state=42
)
```

---

# 13. `alpha` in MultiTaskLasso

`alpha` controls regularization strength.

Generally:

```text
alpha ↑
→ stronger regularization
```

and:

```text
alpha ↓
→ weaker regularization
```

Example:

```python
MultiTaskLasso(alpha=0.01)
```

has weaker regularization than:

```python
MultiTaskLasso(alpha=10)
```

---

# 14. What Happens When Alpha is Too Small?

Very small alpha:

```text
Regularization ↓
       ↓
Model becomes more flexible
       ↓
Potential overfitting
```

Example:

```python
model = MultiTaskLasso(alpha=0.0001)
```

---

# 15. What Happens When Alpha is Too Large?

Very large alpha:

```text
Regularization ↑
       ↓
Coefficients strongly constrained
       ↓
Potential underfitting
```

Example:

```python
model = MultiTaskLasso(alpha=100)
```

---

# 16. `fit_intercept`

Controls whether the model estimates an intercept.

Default behavior:

```python
fit_intercept=True
```

Example:

```python
model = MultiTaskLasso(
    alpha=0.1,
    fit_intercept=True
)
```

---

# 17. `max_iter`

Maximum number of optimization iterations.

Example:

```python
model = MultiTaskLasso(
    alpha=0.1,
    max_iter=10000
)
```

If optimization does not converge, increasing `max_iter` can help.

However, convergence problems may also indicate that the data should be scaled.

---

# 18. `tol`

Tolerance for optimization.

Example:

```python
model = MultiTaskLasso(
    alpha=0.1,
    tol=1e-4
)
```

Smaller tolerance generally means stricter convergence criteria.

---

# 19. `selection`

Controls the coordinate selection strategy during optimization.

Common options:

```text
cyclic
random
```

Example:

```python
model = MultiTaskLasso(
    alpha=0.1,
    selection="cyclic"
)
```

or:

```python
model = MultiTaskLasso(
    alpha=0.1,
    selection="random",
    random_state=42
)
```

---

# 20. MultiTaskElasticNet

## 20.1 What is MultiTaskElasticNet?

`MultiTaskElasticNet` extends MultiTaskLasso by combining:

```text
L1 regularization
+
L2 regularization
```

for multiple related targets.

```python
from sklearn.linear_model import MultiTaskElasticNet
```

Basic syntax:

```python
model = MultiTaskElasticNet(
    alpha=1.0,
    l1_ratio=0.5
)
```

---

# 21. ElasticNet Review

ElasticNet combines:

```text
L1 + L2
```

L1 encourages sparsity:

```text
Some coefficients → 0
```

L2 helps stabilize coefficients, especially with correlated features.

---

# 22. MultiTaskElasticNet Concept

Suppose:

```text
X
 ↓
MultiTaskElasticNet
 ↓
 ├── Target 1
 ├── Target 2
 └── Target 3
```

It jointly learns all targets while using a combination of:

```text
L1-style group sparsity
+
L2-style shrinkage
```

---

# 23. MultiTaskElasticNet Mathematical Form

Conceptually, the objective contains:

```text
Prediction Loss
+
α × l1_ratio × Group L1
+
α × (1-l1_ratio) × Group L2
```

The exact matrix formulation uses norms over the coefficient matrix.

The important interpretation is:

```text
alpha
    ↓
overall regularization strength

l1_ratio
    ↓
balance between L1 and L2
```

---

# 24. Understanding `l1_ratio`

`l1_ratio` controls the L1/L2 balance.

```text
l1_ratio = 1
→ mostly/all L1 component

l1_ratio = 0.5
→ balanced mixture

l1_ratio = 0.1
→ mostly L2 component
```

Example:

```python
MultiTaskElasticNet(
    alpha=0.1,
    l1_ratio=0.9
)
```

uses a stronger L1 component than:

```python
MultiTaskElasticNet(
    alpha=0.1,
    l1_ratio=0.1
)
```

---

# 25. MultiTaskElasticNet Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskElasticNet
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

model = MultiTaskElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```

---

# 26. Checking MultiTaskElasticNet Coefficients

```python
print(model.coef_)
```

Example shape:

```text
(3, 20)
```

means:

```text
3 target variables
20 input features
```

---

# 27. MultiTaskElasticNet Parameters

Important parameters:

```text
alpha
l1_ratio
fit_intercept
max_iter
tol
selection
random_state
```

Example:

```python
model = MultiTaskElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    max_iter=10000,
    tol=1e-4,
    random_state=42
)
```

---

# 28. MultiTaskLasso vs MultiTaskElasticNet

| Feature                 | MultiTaskLasso   | MultiTaskElasticNet |
| ----------------------- | ---------------- | ------------------- |
| Multiple targets        | Yes              | Yes                 |
| L1-style regularization | Yes              | Yes                 |
| L2 component            | No               | Yes                 |
| Feature/group selection | Yes              | Yes                 |
| Main parameters         | `alpha`          | `alpha`, `l1_ratio` |
| Correlated features     | Can be difficult | Often more stable   |
| Sparse solution         | Yes              | Often yes           |

---

# 29. Lasso vs MultiTaskLasso

### Lasso

```text
Single target:

X → y
```

### MultiTaskLasso

```text
Multiple targets:

X → y₁
  → y₂
  → y₃
```

The key difference is that MultiTaskLasso learns related targets jointly.

---

# 30. ElasticNet vs MultiTaskElasticNet

### ElasticNet

```text
X → y
```

Single-output regression.

### MultiTaskElasticNet

```text
X → y₁
  → y₂
  → y₃
```

Multiple related regression targets.

---

# 31. Shared Feature Selection

This is one of the most important ideas.

Suppose there are:

```text
10 features
3 tasks
```

A MultiTask model can encourage:

```text
Feature 1 → useful for Task 1, 2, 3
Feature 2 → not useful
Feature 3 → useful for Task 1, 2, 3
```

This is called **joint/shared feature selection**.

---

# 32. Why Is Shared Feature Selection Useful?

Imagine predicting:

```text
Task 1 → Temperature
Task 2 → Humidity
Task 3 → Pressure
```

Input features:

```text
wind speed
air pressure
cloud coverage
time
location
season
```

Several tasks may depend on the same features.

A MultiTask model can exploit this relationship.

---

# 33. When Should You Use MultiTaskLasso?

Use `MultiTaskLasso` when:

* You have multiple regression targets
* The targets are related
* Shared feature selection is desirable
* You want sparse coefficients
* You prefer L1-style regularization

---

# 34. When Should You Use MultiTaskElasticNet?

Use `MultiTaskElasticNet` when:

* You have multiple related targets
* Features are correlated
* You want L1-style sparsity
* You also want L2 stabilization
* You need a compromise between Lasso and Ridge-like behavior

---

# 35. Data Shape

This is extremely important.

For:

```text
MultiTaskLasso
MultiTaskElasticNet
```

normally:

```text
X → 2D
y → 2D
```

Example:

```python
print(X.shape)
print(y.shape)
```

Output:

```text
(300, 20)
(300, 3)
```

Meaning:

```text
300 samples
20 features
3 targets
```

---

# 36. Single Target Shape vs Multi-Task Shape

Single-output regression:

```text
X.shape = (300, 20)
y.shape = (300,)
```

Multi-output regression:

```text
X.shape = (300, 20)
y.shape = (300, 3)
```

Therefore:

```text
Lasso
→ y is usually 1D

MultiTaskLasso
→ y is 2D
```

---

# 37. StandardScaler with Multi-Task Models

Scaling can be useful for linear models with regularization.

Example:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLasso

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", MultiTaskLasso(alpha=0.1))
])
```

Then:

```python
model.fit(X_train, y_train)
```

---

# 38. Complete MultiTaskLasso Pipeline

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskLasso
from sklearn.metrics import mean_squared_error

X, y = make_regression(
    n_samples=500,
    n_features=30,
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
    ("regressor", MultiTaskLasso(
        alpha=0.1,
        max_iter=10000
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(
    "MSE:",
    mean_squared_error(y_test, y_pred)
)
```

---

# 39. Complete MultiTaskElasticNet Pipeline

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import mean_squared_error

X, y = make_regression(
    n_samples=500,
    n_features=30,
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
    ("regressor", MultiTaskElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=10000
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(
    "MSE:",
    mean_squared_error(y_test, y_pred)
)
```

---

# 40. Evaluating Multi-Task Regression

For multiple targets, you can calculate MSE in different ways.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(
    y_test,
    y_pred
)

print(mse)
```

By default, the errors are aggregated across outputs.

You can also inspect each target separately:

```python
mse_each = mean_squared_error(
    y_test,
    y_pred,
    multioutput="raw_values"
)

print(mse_each)
```

Example:

```text
Target 1 MSE = 120
Target 2 MSE = 95
Target 3 MSE = 150
```

---

# 41. MAE for Multi-Task Regression

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(
    y_test,
    y_pred,
    multioutput="raw_values"
)

print(mae)
```

This returns an error value for each target.

---

# 42. R² for Multi-Task Regression

```python
from sklearn.metrics import r2_score

r2 = r2_score(
    y_test,
    y_pred,
    multioutput="raw_values"
)

print(r2)
```

Example:

```text
Target 1 → 0.91
Target 2 → 0.87
Target 3 → 0.89
```

---

# 43. Multi-Task Model Workflow

```text
Dataset
   ↓
Multiple Related Targets
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
MultiTask Model
   ↓
Fit
   ↓
Predict Multiple Targets
   ↓
Evaluate Each Target
```

---

# 44. MultiTaskLasso Workflow

```text
X + Y
 ↓
MultiTaskLasso
 ↓
L1-style group regularization
 ↓
Shared sparsity
 ↓
Multiple predictions
```

---

# 45. MultiTaskElasticNet Workflow

```text
X + Y
 ↓
MultiTaskElasticNet
 ↓
L1 + L2 group regularization
 ↓
Shared sparsity + shrinkage
 ↓
Multiple predictions
```

---

# 46. What Does `coef_` Mean?

For MultiTask models:

```python
model.coef_
```

contains the learned coefficients for all tasks.

Conceptually:

```text
                Features
             F1    F2    F3    F4

Task 1       2.1   0     1.4   0
Task 2       1.8   0     1.2   0
Task 3       2.0   0     1.5   0
```

Here Feature 2 and Feature 4 have been eliminated across tasks.

---

# 47. Intercept

You can access:

```python
model.intercept_
```

Example:

```python
print(model.intercept_)
```

For multiple targets, the intercept is typically a vector containing one intercept per target.

---

# 48. Prediction

Use:

```python
y_pred = model.predict(X_test)
```

For three targets:

```text
Input sample
     ↓
Model
     ↓
[Prediction 1, Prediction 2, Prediction 3]
```

Example:

```text
[250000, 18000, 5000]
```

could represent:

```text
price
rent
tax
```

---

# 49. MultiTaskLasso Advantages

Advantages:

1. Handles multiple regression targets.
2. Learns related tasks jointly.
3. Encourages shared feature selection.
4. Produces sparse solutions.
5. Can reduce redundant modeling.
6. Useful when tasks share predictive structure.

---

# 50. MultiTaskLasso Disadvantages

Disadvantages:

1. Assumes tasks have useful relationships.
2. Can perform poorly if tasks are unrelated.
3. L1 regularization can be sensitive to correlated features.
4. Requires appropriate target shape.
5. Scaling may be important.
6. Hyperparameter tuning may be necessary.

---

# 51. MultiTaskElasticNet Advantages

Advantages:

1. Handles multiple targets.
2. Combines L1 and L2 regularization.
3. Encourages shared sparsity.
4. Can handle correlated features better than pure L1.
5. Provides a tunable L1/L2 balance.
6. Useful for high-dimensional data.

---

# 52. MultiTaskElasticNet Disadvantages

Disadvantages:

1. More hyperparameters.
2. Can require more computation.
3. Requires related tasks for the multi-task assumption to be useful.
4. Scaling is often important.
5. Choosing `alpha` and `l1_ratio` can require tuning.

---

# 53. MultiTaskLasso vs Separate Lasso Models

Suppose:

```text
3 targets
```

Approach 1:

```text
Lasso Model 1 → Target 1
Lasso Model 2 → Target 2
Lasso Model 3 → Target 3
```

Approach 2:

```text
MultiTaskLasso → Target 1, 2, 3
```

The first approach allows completely independent feature selection.

The second encourages a shared feature-selection structure.

---

# 54. When Should You NOT Use MultiTask Models?

Do not automatically use multi-task learning just because you have multiple targets.

If:

```text
Task 1 and Task 2
```

are completely unrelated, forcing shared structure may hurt performance.

Use multi-task models when there is a reasonable relationship between tasks.

---

# 55. MultiTaskLasso vs MultiTaskElasticNet

Think of them this way:

```text
MultiTaskLasso
       ↓
More L1-focused
       ↓
Strong sparsity
```

while:

```text
MultiTaskElasticNet
       ↓
L1 + L2
       ↓
Sparsity + stabilization
```

---

# 56. Hyperparameter Tuning

You can tune:

### MultiTaskLasso

```text
alpha
```

Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import MultiTaskLasso

model = MultiTaskLasso()

params = {
    "alpha": [0.001, 0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

# 57. Tuning MultiTaskElasticNet

Tune:

```text
alpha
l1_ratio
```

Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import MultiTaskElasticNet

model = MultiTaskElasticNet(
    max_iter=10000
)

params = {
    "alpha": [0.001, 0.01, 0.1, 1],
    "l1_ratio": [0.1, 0.5, 0.9]
}

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

# 58. MultiTaskLasso with Cross-Validation

Instead of manually tuning alpha with GridSearchCV, scikit-learn also provides:

```python
MultiTaskLassoCV
```

Example:

```python
from sklearn.linear_model import MultiTaskLassoCV

model = MultiTaskLassoCV(
    cv=5,
    max_iter=10000
)

model.fit(X_train, y_train)

print(model.alpha_)
```

So:

```text
MultiTaskLasso
→ alpha manually specified

MultiTaskLassoCV
→ alpha selected using CV
```

---

# 59. MultiTaskElasticNet with Cross-Validation

Similarly:

```python
from sklearn.linear_model import MultiTaskElasticNetCV

model = MultiTaskElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    cv=5,
    max_iter=10000
)

model.fit(X_train, y_train)

print(model.alpha_)
print(model.l1_ratio_)
```

Therefore:

```text
MultiTaskElasticNet
→ manually specify alpha/l1_ratio

MultiTaskElasticNetCV
→ automatically search using CV
```

---

# 60. MultiTask Models and Previous Section

Your previous section contained:

```text
MultiTaskLassoCV
MultiTaskElasticNetCV
```

Now:

```text
MultiTaskLasso
MultiTaskElasticNet
```

The difference is Cross-Validation.

| Model                 | CV             |
| --------------------- | -------------- |
| MultiTaskLasso        | No built-in CV |
| MultiTaskLassoCV      | Yes            |
| MultiTaskElasticNet   | No built-in CV |
| MultiTaskElasticNetCV | Yes            |

---

# 61. Full Family Relationship

```text
Lasso
  |
  +── LassoCV

ElasticNet
  |
  +── ElasticNetCV

MultiTaskLasso
  |
  +── MultiTaskLassoCV

MultiTaskElasticNet
  |
  +── MultiTaskElasticNetCV
```

The `CV` versions add built-in cross-validation for hyperparameter selection.

---

# 62. Regularization Family

```text
                    Regularization
                         |
              +----------+----------+
              |                     |
             L1                    L2
              |                     |
           Lasso                  Ridge
              |
              +----------+
                         |
                     ElasticNet
                       L1 + L2
```

For multi-task:

```text
MultiTaskLasso
       ↓
Multi-task L1-style regularization

MultiTaskElasticNet
       ↓
Multi-task L1 + L2 regularization
```

---

# 63. Important Difference: Multi-Output vs Multi-Task

These terms are related but not always identical in every machine learning context.

### Multi-output regression

Simply means:

```text
Multiple outputs
```

### Multi-task learning

Means:

```text
Multiple tasks
+
Joint learning
+
Shared information/structure
```

MultiTaskLasso and MultiTaskElasticNet specifically impose joint regularization across the targets.

---

# 64. Practical Example

Suppose a company wants to predict:

```text
Input:
customer age
income
purchase history
website activity
location

Targets:
product A spending
product B spending
product C spending
```

The targets may be related because customer behavior affects all product categories.

A MultiTask model can learn:

```text
Customer Data
       ↓
MultiTaskElasticNet
       ↓
 ┌─────┼─────┐
 ↓     ↓     ↓
A      B      C
```

---

# 65. Another Example: Sensor Data

Suppose a machine has sensors:

```text
temperature
pressure
vibration
speed
voltage
```

We want to predict:

```text
motor temperature
motor wear
energy consumption
```

These outputs may be related.

A MultiTask model can learn the shared predictive structure.

---

# 66. Interview Questions — Beginner

### Q1. What is MultiTaskLasso?

A multi-output linear regression model using joint L1-style regularization.

### Q2. What is MultiTaskElasticNet?

A multi-output linear regression model combining L1 and L2 regularization.

### Q3. What is multi-task learning?

Learning multiple related prediction tasks jointly.

### Q4. What is the main difference between Lasso and MultiTaskLasso?

Lasso is generally single-output; MultiTaskLasso jointly handles multiple outputs.

### Q5. What does alpha control?

The strength of regularization.

### Q6. What does l1_ratio control?

The L1/L2 balance in MultiTaskElasticNet.

### Q7. What shape should y have?

For multi-task regression, `y` is generally 2D:

```text
(n_samples, n_targets)
```

---

# 67. Interview Questions — Intermediate

### Q1. Why use MultiTaskLasso instead of separate Lasso models?

Because it can exploit shared structure and encourage common feature selection across related tasks.

### Q2. Why use MultiTaskElasticNet instead of MultiTaskLasso?

The L2 component can improve behavior with correlated features while maintaining L1-style sparsity.

### Q3. What happens when alpha increases?

Regularization becomes stronger.

### Q4. What happens when l1_ratio increases?

The L1 component becomes relatively stronger.

### Q5. Can MultiTaskLasso perform feature selection?

Yes. Its joint L1-style regularization can set entire feature groups across tasks to zero.

### Q6. Why should tasks be related?

Because multi-task learning assumes that sharing information between tasks can be useful.

---

# 68. Interview Questions — Advanced

### Q1. Why is the regularization called group-wise?

Because coefficients corresponding to the same feature across multiple tasks are treated as a group.

Conceptually:

```text
Feature i:
[wᵢ₁, wᵢ₂, wᵢ₃]
```

is regularized as a group.

### Q2. What is the main advantage of joint sparsity?

It can identify a common subset of predictive features across related tasks.

### Q3. What happens if tasks are unrelated?

Shared regularization may become a disadvantage and can reduce performance.

### Q4. Why can ElasticNet be useful with correlated features?

The L2 component helps distribute/shrink correlated coefficients more smoothly than pure L1 regularization.

### Q5. Difference between MultiTaskLasso and MultiTaskLassoCV?

```text
MultiTaskLasso
→ alpha specified manually

MultiTaskLassoCV
→ alpha selected using Cross-Validation
```

### Q6. Difference between MultiTaskElasticNet and MultiTaskElasticNetCV?

```text
MultiTaskElasticNet
→ alpha and l1_ratio specified manually

MultiTaskElasticNetCV
→ selected using Cross-Validation
```

---

# 69. Quick Revision

```text
MultiTaskLasso
→ Multi-output regression
→ Joint L1-style regularization
→ Shared sparsity
→ alpha controls regularization
```

```text
MultiTaskElasticNet
→ Multi-output regression
→ L1 + L2
→ Shared sparsity + shrinkage
→ alpha + l1_ratio
```

---

# 70. Final Comparison

| Model                 | Targets  | Regularization | Feature Selection | CV  |
| --------------------- | -------- | -------------- | ----------------- | --- |
| Lasso                 | Single   | L1             | Yes               | No  |
| LassoCV               | Single   | L1             | Yes               | Yes |
| MultiTaskLasso        | Multiple | Joint L1-style | Yes               | No  |
| MultiTaskLassoCV      | Multiple | Joint L1-style | Yes               | Yes |
| ElasticNet            | Single   | L1 + L2        | Yes               | No  |
| ElasticNetCV          | Single   | L1 + L2        | Yes               | Yes |
| MultiTaskElasticNet   | Multiple | Joint L1 + L2  | Yes               | No  |
| MultiTaskElasticNetCV | Multiple | Joint L1 + L2  | Yes               | Yes |

---

# 71. One-Line Cheat Sheet

```text
Lasso
→ One target + L1

MultiTaskLasso
→ Multiple targets + joint L1-style regularization

ElasticNet
→ One target + L1 + L2

MultiTaskElasticNet
→ Multiple targets + joint L1 + L2
```

And:

```text
CV suffix
→ Built-in Cross-Validation
```

Therefore:

```text
MultiTaskLasso
       ↓
MultiTaskLassoCV

MultiTaskElasticNet
       ↓
MultiTaskElasticNetCV
```

---

# 72. Final Summary

The two main models in this section are:

## MultiTaskLasso

```text
Multiple related targets
        +
Joint L1-style regularization
        +
Shared feature selection
```

Use it when you want a sparse multi-output model and believe the tasks share predictive features.

## MultiTaskElasticNet

```text
Multiple related targets
        +
L1 regularization
        +
L2 regularization
        +
Shared feature structure
```

Use it when you want the sparsity of L1 together with the stabilizing effect of L2.

### Most Important Formulas

```text
MultiTaskLasso:

Loss + α × Group-L1 penalty
```

```text
MultiTaskElasticNet:

Loss
+ α × l1_ratio × Group-L1
+ α × (1-l1_ratio) × Group-L2
```

### Most Important Parameters

```text
MultiTaskLasso:
    alpha
    max_iter
    tol
    selection

MultiTaskElasticNet:
    alpha
    l1_ratio
    max_iter
    tol
    selection
```

### Most Important Concept

```text
Single-task:

X → y


Multi-task:

X → y₁
  → y₂
  → y₃
```

The key benefit is **joint learning of related tasks with shared regularization**.
