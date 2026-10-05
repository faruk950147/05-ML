# Classification Models — Full Notes

## Beginner → Intermediate → Advanced

---

# 1. Classification Models

The main Scikit-Learn classification models covered in these notes are:

```text
LogisticRegression
SGDClassifier
PassiveAggressiveClassifier
Perceptron
RidgeClassifier
RidgeClassifierCV
```

These are mainly **linear classification models**.

---

# Beginner Level

# 2. What is Classification?

Classification is a **Supervised Machine Learning technique** used to predict a **discrete class/category**.

Examples:

```text
Spam Detection
→ Spam / Not Spam

Disease Detection
→ Disease / No Disease

Student Result
→ Pass / Fail

Loan Approval
→ Approved / Rejected

Image Classification
→ Cat / Dog

Customer Churn
→ Churn / Not Churn
```

Unlike Regression:

```text
Regression
→ Predict continuous numerical values

Classification
→ Predict discrete classes
```

Example:

```text
Hours Studied → Result

2 → Fail
4 → Fail
6 → Pass
8 → Pass
```

Here:

```text
Input  = Hours Studied
Output = Pass / Fail
```

Therefore, this is a Classification problem.

---

# 3. Types of Classification

Classification can be divided into several types.

## 3.1 Binary Classification

There are exactly two classes.

Example:

```text
Spam
Not Spam
```

or:

```text
0
1
```

Examples:

```text
Fraud / Not Fraud
Pass / Fail
Yes / No
Disease / No Disease
```

---

## 3.2 Multiclass Classification

There are more than two classes.

Example:

```text
Cat
Dog
Horse
```

Another example:

```text
0
1
2
3
...
9
```

for handwritten digit classification.

---

## 3.3 Multilabel Classification

One sample can belong to multiple labels simultaneously.

Example:

```text
Movie:
Action
Comedy
Drama
```

A movie may have:

```text
Action = 1
Comedy = 1
Drama = 0
```

Multilabel classification is different from ordinary multiclass classification.

---

# 4. Classification Input and Output

Suppose:

```text
Age
Salary
Credit Score
```

are features.

We want to predict:

```text
Loan Approved
```

Dataset:

```text
Age    Salary    Credit Score    Approved

25     30000     600             No
35     60000     720             Yes
45     80000     760             Yes
23     25000     580             No
```

Therefore:

```text
X = Features

y = Target / Class
```

---

# 5. Feature and Target

## Feature

A feature is an input variable used by the model.

Example:

```text
Age
Salary
Credit Score
```

## Target

The target is the class we want to predict.

Example:

```text
Loan Approved
```

Relationship:

```text
Features
   ↓
Classification Model
   ↓
Predicted Class
```

---

# 6. Class Labels

Classification targets are usually represented using labels.

Example:

```text
No  → 0
Yes → 1
```

For three classes:

```text
Cat   → 0
Dog   → 1
Horse → 2
```

The numerical representation is only a label encoding; the numbers do not automatically imply meaningful numeric distance.

---

# 7. Decision Boundary

A classification model tries to separate different classes.

Example:

```text
Class 0 → ○
Class 1 → ×
```

A simple linear classifier may find a boundary:

```text
○ ○ ○ ○ | × × ×
○ ○ ○ ○ | × × ×
○ ○ ○ ○ | × × ×
          ↑
    Decision Boundary
```

For two features, a linear decision boundary can often be represented as:

```text
w₁x₁ + w₂x₂ + b = 0
```

Where:

```text
w = Model weights
x = Input features
b = Bias / intercept
```

---

# 8. Linear Classification

The six models in this section are primarily linear classifiers.

General form:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

or:

```text
z = wᵀx + b
```

The model then uses this score to determine the class.

Important:

```text
Linear classifier
≠
Linear Regression
```

They may use a similar linear equation, but their training objectives and prediction tasks are different.

---

# 9. Logistic Regression

Despite its name, Logistic Regression is mainly used for **classification**.

It predicts the probability of a class.

For binary classification:

```text
z = wᵀx + b
```

Then the sigmoid function converts the score into a probability.

Formula:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

Therefore:

```text
P(y=1|x) = σ(wᵀx + b)
```

The output is between:

```text
0 and 1
```

---

# 10. Sigmoid Function

The sigmoid function is:

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

Example:

If:

```text
z = 0
```

Then:

```text
σ(0) = 0.5
```

If:

```text
z = large positive value
```

then:

```text
σ(z) → 1
```

If:

```text
z = large negative value
```

then:

```text
σ(z) → 0
```

Conceptually:

```text
z
↓
Linear Score
↓
Sigmoid
↓
Probability
↓
Class
```

---

# 11. Logistic Regression Example

Suppose:

```text
Probability = 0.80
```

For a default threshold of 0.5:

```text
0.80 >= 0.5
```

Therefore:

```text
Predicted Class = 1
```

If:

```text
Probability = 0.20
```

Then:

```text
0.20 < 0.5
```

Therefore:

```text
Predicted Class = 0
```

Important:

```text
The 0.5 threshold is a common default for binary classification,
but it can be changed depending on the application.
```

---

# 12. Logistic Regression Python

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

Prediction:

```python
y_pred = model.predict(X_test)
```

Probability:

```python
y_prob = model.predict_proba(X_test)
```

Example:

```python
print(y_pred)
print(y_prob)
```

---

# 13. Logistic Regression Probability

For binary classification:

```python
model.predict_proba(X_test)
```

may produce:

```text
[
    [0.80, 0.20],
    [0.30, 0.70],
    [0.10, 0.90]
]
```

Each row represents class probabilities.

For example:

```text
[0.30, 0.70]
```

means:

```text
P(Class 0) = 0.30
P(Class 1) = 0.70
```

Prediction:

```text
Class 1
```

---

# 14. Logistic Regression Decision Function

Logistic Regression also provides:

```python
model.decision_function(X_test)
```

The decision score is the linear score before converting it into a probability.

Conceptually:

```text
Features
   ↓
wᵀx + b
   ↓
Decision Score
   ↓
Sigmoid
   ↓
Probability
```

---

# 15. Logistic Regression Loss

Logistic Regression commonly uses **log loss / cross-entropy loss**.

Binary cross-entropy:

```text
Loss =
-[y log(p) + (1-y) log(1-p)]
```

Where:

```text
y = Actual class
p = Predicted probability
```

The model learns parameters that minimize the overall loss, with regularization commonly included.

---

# 16. Logistic Regression Regularization

Logistic Regression supports regularization.

Common types include:

```text
L1
L2
ElasticNet
```

depending on the selected solver/configuration.

Example:

```python
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000
)
```

---

# 17. C Parameter

In Logistic Regression:

```text
C
```

controls the inverse of regularization strength.

Important:

```text
Small C
→ Stronger regularization

Large C
→ Weaker regularization
```

Example:

```python
LogisticRegression(C=0.1)
```

uses stronger regularization than:

```python
LogisticRegression(C=10)
```

---

# 18. Multiclass Logistic Regression

Logistic Regression can also handle multiclass classification.

Example:

```text
Cat
Dog
Horse
```

Conceptually, multiclass logistic regression uses multiple class scores/probabilities.

For a multiclass problem:

```text
Class 0 → Probability
Class 1 → Probability
Class 2 → Probability
```

The predicted class is generally the class with the highest probability.

Example:

```text
Cat   = 0.10
Dog   = 0.75
Horse = 0.15
```

Prediction:

```text
Dog
```

---

# 19. SGDClassifier

`SGDClassifier` is a linear classifier trained using **Stochastic Gradient Descent**.

Python:

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    random_state=42
)

model.fit(X_train, y_train)
```

Prediction:

```python
y_pred = model.predict(X_test)
```

---

# 20. Why SGDClassifier?

SGDClassifier is useful for:

```text
Large datasets
High-dimensional datasets
Sparse data
Online learning
Incremental learning
Fast linear classification
```

It processes training examples incrementally rather than solving the optimization problem using a traditional batch solver.

---

# 21. SGDClassifier Loss

`SGDClassifier` supports different loss functions.

Important examples:

```text
hinge
log_loss
modified_huber
perceptron
squared_hinge
```

For example:

```python
model = SGDClassifier(
    loss="log_loss",
    random_state=42
)
```

This gives an SGD-trained logistic-loss classifier.

Another example:

```python
model = SGDClassifier(
    loss="hinge",
    random_state=42
)
```

This uses the hinge loss associated with a linear SVM-style classifier.

---

# 22. SGDClassifier with Log Loss

Example:

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    loss="log_loss",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)
```

This is conceptually similar to Logistic Regression in its loss objective, but the optimization method is different.

```text
LogisticRegression
→ Solver-based optimization

SGDClassifier(loss="log_loss")
→ Stochastic Gradient Descent
```

---

# 23. SGDClassifier Hyperparameters

Important parameters:

```text
loss
penalty
alpha
max_iter
tol
learning_rate
eta0
early_stopping
class_weight
random_state
```

Example:

```python
model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=0.0001,
    max_iter=1000,
    random_state=42
)
```

---

# 24. Alpha in SGDClassifier

`alpha` controls regularization strength.

Generally:

```text
Higher alpha
→ Stronger regularization

Lower alpha
→ Weaker regularization
```

Example:

```python
SGDClassifier(alpha=0.001)
```

has stronger regularization than:

```python
SGDClassifier(alpha=0.00001)
```

---

# 25. Partial Fit

One important feature of `SGDClassifier` is incremental learning.

Example:

```python
model.partial_fit(
    X_batch,
    y_batch,
    classes=[0, 1]
)
```

This allows the model to learn from batches of data.

Useful for:

```text
Streaming data
Very large datasets
Online learning
Continuous training
```

---

# 26. PassiveAggressiveClassifier

`PassiveAggressiveClassifier` is an **online linear classification algorithm**.

Basic idea:

```text
Prediction is sufficiently correct
→ Stay Passive

Prediction violates the desired condition
→ Become Aggressive and update
```

Therefore:

```text
Correct enough
→ Little/no update

Incorrect / insufficient margin
→ Update model
```

---

# 27. Passive-Aggressive Concept

Suppose the model receives:

```text
Sample 1
Sample 2
Sample 3
Sample 4
...
```

For each sample:

```text
Prediction
   ↓
Is prediction good enough?
   ↓
Yes → No/Small update

No → Aggressive update
```

This makes it useful for online learning.

---

# 28. PassiveAggressiveClassifier Python

```python
from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(
    random_state=42
)

model.fit(X_train, y_train)
```

Prediction:

```python
y_pred = model.predict(X_test)
```

---

# 29. PassiveAggressiveClassifier Hyperparameters

Important parameters include:

```text
C
loss
max_iter
tol
average
class_weight
random_state
```

Example:

```python
model = PassiveAggressiveClassifier(
    C=1.0,
    max_iter=1000,
    random_state=42
)
```

---

# 30. PassiveAggressiveClassifier Loss

Common choices include:

```text
hinge
squared_hinge
```

Example:

```python
model = PassiveAggressiveClassifier(
    loss="hinge",
    random_state=42
)
```

---

# 31. PassiveAggressiveClassifier Partial Fit

It can also be trained incrementally.

```python
model.partial_fit(
    X_batch,
    y_batch,
    classes=[0, 1]
)
```

Useful for:

```text
Online learning
Streaming data
Large datasets
Incremental learning
```

---

# 32. Perceptron

The Perceptron is one of the simplest linear classification algorithms.

Basic idea:

```text
Make prediction
↓
If correct
→ Keep current weights

If incorrect
→ Update weights
```

It is a classic online learning algorithm.

---

# 33. Perceptron Equation

The model calculates:

```text
z = wᵀx + b
```

For binary classification, a sign-based decision can be used conceptually:

```text
z >= 0
→ Class 1

z < 0
→ Class 0
```

The model updates its parameters when a training example is misclassified.

---

# 34. Perceptron Python

```python
from sklearn.linear_model import Perceptron

model = Perceptron(
    random_state=42
)

model.fit(X_train, y_train)
```

Prediction:

```python
y_pred = model.predict(X_test)
```

---

# 35. Perceptron Update Rule

A simplified update rule is:

```text
w_new = w_old + η y x
```

and:

```text
b_new = b_old + η y
```

when the example is misclassified under the chosen label representation.

Where:

```text
η = Learning Rate
y = Target label
x = Feature vector
```

The exact implementation details depend on the label representation and training configuration.

---

# 36. Perceptron Characteristics

Perceptron:

```text
Linear classifier
Fast
Simple
Online learning capable
No probability output by default
```

Important limitation:

```text
A basic Perceptron works well when classes are approximately
linearly separable, but it is not guaranteed to converge on
non-linearly separable data.
```

---

# 37. Perceptron Hyperparameters

Important parameters:

```text
penalty
alpha
max_iter
eta0
early_stopping
class_weight
random_state
```

Example:

```python
model = Perceptron(
    max_iter=1000,
    eta0=1.0,
    random_state=42
)
```

---

# 38. RidgeClassifier

`RidgeClassifier` is a classifier based on **Ridge regression with L2 regularization**.

It learns linear decision scores and uses those scores for classification.

Conceptually:

```text
Ridge Regression
       ↓
Linear Scores
       ↓
Class Decision
```

Python:

```python
from sklearn.linear_model import RidgeClassifier

model = RidgeClassifier()

model.fit(X_train, y_train)
```

Prediction:

```python
y_pred = model.predict(X_test)
```

---

# 39. RidgeClassifier Concept

RidgeClassifier uses an L2-regularized least-squares objective.

Conceptually:

```text
Loss =
Least Squares Loss
+
L2 Penalty
```

L2 penalty:

```text
λ Σ w²
```

This helps control large coefficients.

---

# 40. RidgeClassifier Alpha

Important parameter:

```text
alpha
```

`alpha` controls the regularization strength.

Generally:

```text
Higher alpha
→ Stronger regularization

Lower alpha
→ Weaker regularization
```

Example:

```python
model = RidgeClassifier(alpha=10)
```

---

# 41. RidgeClassifier Prediction

RidgeClassifier does not normally provide class probabilities like Logistic Regression.

Instead, it provides decision scores.

```python
scores = model.decision_function(X_test)
```

Prediction:

```python
y_pred = model.predict(X_test)
```

For multiple classes, the class associated with the highest score is selected.

---

# 42. RidgeClassifierCV

`RidgeClassifierCV` is a version of RidgeClassifier that uses **cross-validation to select the regularization parameter**.

Python:

```python
from sklearn.linear_model import RidgeClassifierCV

model = RidgeClassifierCV(
    alphas=[0.1, 1.0, 10.0, 100.0]
)

model.fit(X_train, y_train)
```

The model evaluates candidate alpha values and selects an appropriate one using cross-validation.

---

# 43. RidgeClassifier vs RidgeClassifierCV

```text
RidgeClassifier
→ You select alpha manually

RidgeClassifierCV
→ Cross-validation selects alpha
```

Example:

```python
RidgeClassifier(alpha=1.0)
```

versus:

```python
RidgeClassifierCV(
    alphas=[0.01, 0.1, 1, 10, 100]
)
```

---

# 44. Feature Scaling

Feature scaling is important for many linear classification models.

Suppose:

```text
Age = 25
Salary = 500000
```

The feature scales are very different.

This can negatively affect optimization, especially for:

```text
SGDClassifier
PassiveAggressiveClassifier
Perceptron
```

Scaling is also commonly useful for regularized linear models such as:

```text
LogisticRegression
RidgeClassifier
```

---

# 45. Standardization

Formula:

```text
z = (x - μ) / σ
```

Where:

```text
μ = Mean
σ = Standard Deviation
```

Python:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Important:

```text
fit_transform()
→ Training data

transform()
→ Test data
```

Never fit the scaler separately on the test set.

---

# 46. Pipeline

A Pipeline combines preprocessing and model training.

Example:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
```

Advantages:

```text
Prevents preprocessing mistakes
Reduces data leakage
Cleaner workflow
Works with cross-validation
Works with hyperparameter tuning
```

---

# 47. Train-Test Split for Classification

Classification should normally use a train/test split.

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Important:

```text
stratify=y
```

helps preserve class proportions between training and testing sets.

---

# 48. Why Stratify?

Suppose the dataset contains:

```text
Class 0 → 90%
Class 1 → 10%
```

A random split could accidentally produce an unusual class distribution.

Using:

```python
stratify=y
```

helps maintain approximately the same class proportions.

---

# 49. Classification Evaluation

Important classification metrics:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC
PR-AUC
Log Loss
Balanced Accuracy
```

Different metrics answer different questions.

---

# 50. Confusion Matrix

A confusion matrix summarizes classification predictions.

For binary classification:

```text
                  Predicted
                0         1

Actual 0       TN        FP

Actual 1       FN        TP
```

Where:

```text
TP = True Positive
TN = True Negative
FP = False Positive
FN = False Negative
```

---

# 51. True Positive

True Positive means:

```text
Actual = Positive
Predicted = Positive
```

Example:

```text
Actual: Fraud
Predicted: Fraud
```

Correct positive prediction.

---

# 52. True Negative

True Negative means:

```text
Actual = Negative
Predicted = Negative
```

Example:

```text
Actual: Not Fraud
Predicted: Not Fraud
```

Correct negative prediction.

---

# 53. False Positive

False Positive means:

```text
Actual = Negative
Predicted = Positive
```

Example:

```text
Actual: Not Fraud
Predicted: Fraud
```

This is also called:

```text
Type I Error
```

---

# 54. False Negative

False Negative means:

```text
Actual = Positive
Predicted = Negative
```

Example:

```text
Actual: Fraud
Predicted: Not Fraud
```

This is also called:

```text
Type II Error
```

---

# 55. Accuracy

Accuracy measures the proportion of correct predictions.

Formula:

```text
Accuracy =
(TP + TN) / (TP + TN + FP + FN)
```

Example:

```text
Total Samples = 100

Correct Predictions = 90
```

Therefore:

```text
Accuracy = 90 / 100
         = 0.90
         = 90%
```

Python:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(accuracy)
```

---

# 56. Problem with Accuracy

Accuracy can be misleading for imbalanced datasets.

Example:

```text
1000 samples

990 → Negative
10  → Positive
```

A model that predicts everything as negative gets:

```text
Accuracy = 99%
```

But:

```text
Positive cases detected = 0
```

Therefore, accuracy alone is not sufficient for many imbalanced problems.

---

# 57. Precision

Precision answers:

```text
Of all predicted positives,
how many were actually positive?
```

Formula:

```text
Precision =
TP / (TP + FP)
```

Example:

```text
TP = 80
FP = 20
```

Therefore:

```text
Precision = 80 / (80 + 20)

          = 0.80
```

Precision = 80%

---

# 58. Recall

Recall answers:

```text
Of all actual positives,
how many did the model detect?
```

Formula:

```text
Recall =
TP / (TP + FN)
```

Example:

```text
TP = 80
FN = 20
```

Therefore:

```text
Recall = 80 / (80 + 20)

       = 0.80
```

Recall = 80%

---

# 59. Precision vs Recall

```text
Precision
→ Focuses on predicted positives

Recall
→ Focuses on actual positives
```

High precision:

```text
Few false positives
```

High recall:

```text
Few false negatives
```

Example:

```text
Spam Detection
→ Precision can be important

Disease Screening
→ Recall can be especially important
```

The correct metric depends on the application.

---

# 60. F1 Score

F1 Score is the harmonic mean of Precision and Recall.

Formula:

```text
F1 =
2 × Precision × Recall
-----------------------
Precision + Recall
```

Example:

```text
Precision = 0.8
Recall = 0.6
```

Then:

```text
F1 =
2 × 0.8 × 0.6
--------------
0.8 + 0.6

= 0.96 / 1.4

≈ 0.686
```

Python:

```python
from sklearn.metrics import f1_score

f1 = f1_score(
    y_test,
    y_pred
)

print(f1)
```

---

# 61. Classification Report

Scikit-Learn provides:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

Example output:

```text
              precision    recall    f1-score

Class 0          0.90       0.95       0.92
Class 1          0.80       0.70       0.75
```

It can also show:

```text
accuracy
macro avg
weighted avg
```

---

# 62. Macro Average

Macro average calculates the metric independently for each class and then takes the simple average.

Example:

```text
Class 0 F1 = 0.90
Class 1 F1 = 0.60
Class 2 F1 = 0.80
```

Macro F1:

```text
(0.90 + 0.60 + 0.80) / 3
```

Therefore:

```text
Macro average
→ Gives equal importance to each class
```

---

# 63. Weighted Average

Weighted average considers the number of samples in each class.

Therefore:

```text
Large classes
→ Have greater influence
```

Useful when class sizes are different, but it can hide poor performance on minority classes.

---

# 64. Balanced Accuracy

Balanced Accuracy is useful for imbalanced classification.

For binary classification:

```text
Balanced Accuracy =
(Recall of Class 0 + Recall of Class 1) / 2
```

Python:

```python
from sklearn.metrics import balanced_accuracy_score

score = balanced_accuracy_score(
    y_test,
    y_pred
)

print(score)
```

---

# 65. ROC Curve

ROC means:

```text
Receiver Operating Characteristic
```

It plots:

```text
True Positive Rate
vs
False Positive Rate
```

Where:

```text
TPR = TP / (TP + FN)

FPR = FP / (FP + TN)
```

The threshold is varied to produce different TPR/FPR pairs.

---

# 66. ROC-AUC

AUC means:

```text
Area Under the Curve
```

ROC-AUC measures how well the model ranks positive examples above negative examples.

Conceptually:

```text
AUC ≈ 1
→ Very good ranking

AUC ≈ 0.5
→ Similar to random ranking

AUC < 0.5
→ Worse than random ranking
```

Python:

```python
from sklearn.metrics import roc_auc_score

score = roc_auc_score(
    y_test,
    y_prob
)

print(score)
```

For binary classification, `y_prob` should normally contain the positive-class probability.

---

# 67. Precision-Recall Curve

Precision-Recall curves show the trade-off between:

```text
Precision
Recall
```

They can be especially informative for highly imbalanced datasets.

Example:

```text
Fraud Detection
```

If fraud cases are very rare:

```text
ROC-AUC may look good
```

while:

```text
Precision-Recall
```

can provide a more informative view of positive-class performance.

---

# 68. Log Loss

Log Loss evaluates the quality of predicted probabilities.

Binary formula:

```text
Log Loss =
-[y log(p) + (1-y)log(1-p)]
```

A confident correct prediction gets a small loss.

A confident incorrect prediction gets a large loss.

Python:

```python
from sklearn.metrics import log_loss

loss = log_loss(
    y_test,
    y_prob
)

print(loss)
```

Lower is better.

---

# 69. Which Models Provide Probability?

Among these six models:

```text
LogisticRegression
→ predict_proba()
```

`SGDClassifier` can provide probabilities for compatible losses such as:

```text
log_loss
modified_huber
```

Other models such as:

```text
PassiveAggressiveClassifier
Perceptron
RidgeClassifier
```

normally provide decision scores rather than calibrated probabilities.

For probability estimates from models without native probability output, probability calibration can be considered.

---

# 70. Decision Function

Several linear classifiers provide:

```python
model.decision_function(X_test)
```

This produces a confidence-like decision score.

Examples:

```text
SGDClassifier
PassiveAggressiveClassifier
Perceptron
RidgeClassifier
LogisticRegression
```

The exact interpretation of the score depends on the model.

Important:

```text
Decision score
≠
Probability
```

Do not automatically interpret a decision score such as:

```text
2.4
```

as:

```text
2.4 probability
```

---

# 71. Class Imbalance

Class imbalance occurs when one class has significantly more samples than another.

Example:

```text
Class 0 = 95%
Class 1 = 5%
```

This is an imbalanced dataset.

Problems:

```text
Accuracy can become misleading
Minority class may be ignored
Recall may become poor
```

---

# 72. Handling Class Imbalance

Possible approaches:

```text
class_weight
Oversampling
Undersampling
Threshold adjustment
Appropriate evaluation metrics
```

Example:

```python
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

`class_weight="balanced"` automatically gives more importance to underrepresented classes according to the class frequencies.

---

# 73. Threshold Tuning

For probability-based classifiers, the default threshold is not always optimal.

Example:

```text
Probability = 0.45
```

Default threshold:

```text
0.5
```

would classify it as:

```text
Class 0
```

But if we lower the threshold:

```text
0.4
```

then:

```text
0.45 >= 0.4
```

so it becomes:

```text
Class 1
```

Lowering the threshold can increase recall, but may also increase false positives.

---

# 74. Overfitting in Classification

Overfitting occurs when the model performs very well on training data but poorly on unseen data.

Example:

```text
Training Accuracy = 99%
Testing Accuracy  = 70%
```

This may indicate overfitting.

Solutions:

```text
Regularization
More training data
Feature selection
Cross-validation
Simpler model
Hyperparameter tuning
```

---

# 75. Underfitting in Classification

Underfitting occurs when the model is too simple to capture the underlying pattern.

Example:

```text
Training Accuracy = 65%
Testing Accuracy  = 63%
```

Both are poor.

Solutions:

```text
Better features
More suitable model
Reduce excessive regularization
Feature engineering
```

---

# 76. Bias and Variance

High Bias:

```text
Underfitting
```

High Variance:

```text
Overfitting
```

Goal:

```text
Balance Bias and Variance
```

---

# Intermediate Level

# 77. Regularization in Classification

Regularization helps control model complexity.

Main types:

```text
L1
L2
ElasticNet
```

Conceptually:

```text
Original Loss
+
Regularization Penalty
```

---

# 78. L1 Regularization

L1 penalty:

```text
λ Σ|w|
```

L1 regularization can drive some coefficients exactly to zero.

Therefore:

```text
L1
→ Can perform feature selection
```

Used by compatible models/configurations such as:

```text
LogisticRegression
SGDClassifier
Perceptron
```

---

# 79. L2 Regularization

L2 penalty:

```text
λ Σw²
```

It tends to shrink coefficients toward zero without generally forcing them exactly to zero.

Commonly used in:

```text
LogisticRegression
SGDClassifier
RidgeClassifier
```

---

# 80. LogisticRegression vs RidgeClassifier

```text
LogisticRegression
→ Logistic / log-loss objective
→ Probability output available
→ Strong choice when probability estimates are needed

RidgeClassifier
→ Ridge / least-squares style objective
→ L2 regularization
→ Decision scores
→ No native probability output
```

---

# 81. LogisticRegression vs SGDClassifier

```text
LogisticRegression
→ Optimizes classification objective using a solver
→ Usually convenient for small/medium datasets
→ Predict probabilities

SGDClassifier
→ Uses stochastic gradient descent
→ Suitable for large-scale learning
→ Supports incremental learning
→ Multiple loss functions
```

If:

```python
SGDClassifier(loss="log_loss")
```

then it uses a logistic-loss objective, but the optimization approach is SGD.

---

# 82. SGDClassifier vs Perceptron

```text
SGDClassifier
→ General SGD framework
→ Multiple loss functions
→ Regularization
→ More flexible

Perceptron
→ Classic perceptron learning rule
→ Simple
→ Updates based on classification mistakes
```

Therefore:

```text
Perceptron
→ Simpler algorithm

SGDClassifier
→ More general framework
```

---

# 83. SGDClassifier vs PassiveAggressiveClassifier

```text
SGDClassifier
→ Optimizes a selected loss using SGD

PassiveAggressiveClassifier
→ Updates strongly when the prediction violates
  the desired condition
```

Both can support online/incremental learning.

---

# 84. Perceptron vs Passive Aggressive

```text
Perceptron
→ Update mainly when sample is misclassified

Passive Aggressive
→ Update when sample violates the required margin/loss condition
→ Can make a larger corrective update
```

---

# 85. RidgeClassifier vs RidgeClassifierCV

```text
RidgeClassifier
→ Manual alpha

RidgeClassifierCV
→ Cross-validation selects alpha
```

Example:

```python
model = RidgeClassifier(alpha=1.0)
```

versus:

```python
model = RidgeClassifierCV(
    alphas=[0.01, 0.1, 1, 10, 100]
)
```

---

# 86. Cross Validation

Cross-validation evaluates the model across multiple validation splits.

Example:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print(scores)
print(scores.mean())
```

Example:

```text
[0.90, 0.88, 0.92, 0.89, 0.91]
```

Mean:

```text
0.90
```

---

# 87. Stratified K-Fold

For classification, **StratifiedKFold** is commonly preferred because it attempts to preserve class proportions in each fold.

Example:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Then:

```python
scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1"
)
```

---

# 88. Hyperparameters

Hyperparameters are configuration values selected before/during model selection rather than learned as ordinary model coefficients.

Examples:

```text
LogisticRegression
→ C
→ penalty
→ solver

SGDClassifier
→ loss
→ alpha
→ penalty
→ learning_rate
→ max_iter

PassiveAggressiveClassifier
→ C
→ loss
→ max_iter

Perceptron
→ alpha
→ eta0
→ penalty
→ max_iter

RidgeClassifier
→ alpha

RidgeClassifierCV
→ alphas
→ cv
```

---

# 89. GridSearchCV

GridSearchCV searches through specified hyperparameter combinations.

Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

params = {
    "C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    params,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
```

---

# 90. RandomizedSearchCV

RandomizedSearchCV samples combinations from the search space rather than exhaustively evaluating every combination.

Example:

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier

params = {
    "alpha": [0.00001, 0.0001, 0.001, 0.01],
    "loss": [
        "hinge",
        "log_loss",
        "modified_huber"
    ]
}

search = RandomizedSearchCV(
    SGDClassifier(random_state=42),
    params,
    n_iter=5,
    cv=5,
    random_state=42
)

search.fit(X_train, y_train)
```

---

# 91. Data Leakage

Data leakage occurs when information that should not be available during prediction enters the training process.

Bad:

```text
Entire Dataset
      ↓
Scaling
      ↓
Train-Test Split
```

This allows test-set information to influence scaling.

Correct:

```text
Dataset
   ↓
Train-Test Split
   ↓
Fit preprocessing on Training Data
   ↓
Transform Training Data
   ↓
Transform Test Data
```

Pipeline is usually the safest approach.

---

# 92. Classification Pipeline

Example:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
```

---

# 93. Sparse Data

Many classification problems produce sparse feature matrices.

Examples:

```text
Text Classification
Document Classification
Spam Detection
```

For example:

```text
TF-IDF
Bag of Words
```

can produce high-dimensional sparse matrices.

Linear classifiers are often effective for this type of data.

Examples:

```text
LogisticRegression
SGDClassifier
PassiveAggressiveClassifier
RidgeClassifier
```

---

# 94. Text Classification Example

Suppose:

```text
"I love this movie"
"I hate this movie"
```

We want:

```text
Positive
Negative
```

Pipeline:

```text
Text
 ↓
TF-IDF
 ↓
Linear Classifier
 ↓
Prediction
```

Example:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train_text, y_train)

predictions = model.predict(X_test_text)
```

---

# 95. Online Learning

Online learning means the model can learn incrementally as new data arrives.

Concept:

```text
New Data
   ↓
Model Update
   ↓
New Data
   ↓
Model Update
```

Useful for:

```text
Streaming data
Real-time systems
Large datasets
Continuous learning
```

Models in this group that support incremental learning include:

```text
SGDClassifier
PassiveAggressiveClassifier
Perceptron
```

using `partial_fit()`.

---

# 96. Partial Fit Example

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    loss="log_loss",
    random_state=42
)

classes = [0, 1]

for X_batch, y_batch in batches:

    model.partial_fit(
        X_batch,
        y_batch,
        classes=classes
    )
```

Important:

```text
For the first partial_fit call,
provide the complete list of possible classes.
```

---

# 97. Multiclass Strategy

Some binary classifiers can be extended to multiclass classification.

Common strategies include:

```text
One-vs-Rest (OvR)
One-vs-One (OvO)
```

---

# 98. One-vs-Rest

One-vs-Rest means:

```text
For N classes:

Train N binary classifiers
```

Example:

```text
Classes:

Cat
Dog
Horse
```

Models:

```text
Model 1:
Cat vs Not Cat

Model 2:
Dog vs Not Dog

Model 3:
Horse vs Not Horse
```

Prediction:

```text
Select the class with the strongest score.
```

---

# 99. One-vs-One

One-vs-One trains one classifier for every pair of classes.

For:

```text
3 classes
```

we get:

```text
Cat vs Dog
Cat vs Horse
Dog vs Horse
```

Number of classifiers:

```text
K(K-1)/2
```

Where:

```text
K = Number of classes
```

For 3 classes:

```text
3(3-1)/2
= 3
```

---

# 100. Calibration

A classifier's decision scores are not automatically well-calibrated probabilities.

Calibration asks:

```text
When a model says probability = 0.8,
do roughly 80% of those examples actually belong
to the positive class?
```

Scikit-Learn provides:

```python
from sklearn.calibration import CalibratedClassifierCV
```

Example:

```python
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

base_model = SGDClassifier(
    loss="hinge",
    random_state=42
)

model = CalibratedClassifierCV(
    base_model,
    cv=5
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)
```

---

# Advanced Level

# 101. Logistic Regression Mathematical Form

Linear score:

```text
z = wᵀx + b
```

Sigmoid:

```text
p = 1 / (1 + e⁻ᶻ)
```

Therefore:

```text
p = P(y=1|x)
```

Prediction:

```text
p >= threshold
→ Class 1

p < threshold
→ Class 0
```

---

# 102. Log-Odds

Logistic Regression can also be represented using log-odds.

Odds:

```text
Odds = p / (1-p)
```

Log-odds:

```text
log(p / (1-p))
```

Logistic Regression assumes:

```text
log(p / (1-p))
=
wᵀx + b
```

Therefore:

```text
Log-odds
→ Linear function of features
```

---

# 103. Logistic Regression Coefficient Interpretation

Suppose:

```text
Coefficient = 0.5
```

For a one-unit increase in the feature, holding other features constant:

```text
Log-odds increase by 0.5
```

The corresponding odds multiplier is:

```text
e^0.5
≈ 1.65
```

Therefore, the odds are multiplied by approximately:

```text
1.65
```

for a one-unit increase in that feature, holding other features constant.

Important:

```text
Coefficient is not directly a probability increase.
```

---

# 104. Odds Ratio

For Logistic Regression:

```text
Odds Ratio = e^β
```

Example:

```text
β = 0.693
```

Then:

```text
e^0.693 ≈ 2
```

Interpretation:

```text
A one-unit increase in the feature
multiplies the odds by approximately 2,
holding other features constant.
```

---

# 105. Decision Boundary of Logistic Regression

The decision boundary for a threshold of 0.5 occurs when:

```text
p = 0.5
```

Since:

```text
σ(z) = 0.5
```

when:

```text
z = 0
```

Therefore:

```text
wᵀx + b = 0
```

This is a linear decision boundary.

---

# 106. Linear Decision Boundary

For two features:

```text
w₁x₁ + w₂x₂ + b = 0
```

This produces a line.

For three features:

```text
w₁x₁ + w₂x₂ + w₃x₃ + b = 0
```

This produces a plane.

For higher dimensions:

```text
Hyperplane
```

---

# 107. Nonlinear Classification

These models are primarily linear classifiers.

If the data has a strongly nonlinear boundary:

```text
Linear Model
→ May underfit
```

Possible solutions:

```text
Polynomial Features
Kernel Methods
Tree-Based Models
Random Forest
Gradient Boosting
Neural Networks
```

Example:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

---

# 108. Regularization and Feature Scaling

For regularized linear models, feature scale matters.

Suppose:

```text
Feature A:
0 → 1

Feature B:
0 → 1,000,000
```

The regularization penalty may affect coefficients differently because the feature scales are different.

Therefore:

```text
Scaling
+
Regularization
```

is commonly a good combination.

Example:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

---

# 109. Multicollinearity in Classification

Multicollinearity occurs when features contain highly overlapping information.

Example:

```text
Age in Years
Age in Months
```

These features are highly correlated.

Problems:

```text
Unstable coefficients
Difficult interpretation
Redundant information
```

Regularization can help stabilize linear models.

---

# 110. Confusion Matrix Python

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
```

Example:

```text
[[90 10]
 [ 5 95]]
```

Interpretation:

```text
TN = 90
FP = 10
FN = 5
TP = 95
```

---

# 111. Classification Metrics Python

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("Confusion Matrix:")
print(cm)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 112. Complete Logistic Regression Example

Dataset:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target
```

Split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])
```

Train:

```python
model.fit(
    X_train,
    y_train
)
```

Prediction:

```python
y_pred = model.predict(
    X_test
)
```

Probability:

```python
y_prob = model.predict_proba(
    X_test
)[:, 1]
```

Evaluation:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(y_test, y_pred)
)

print(
    "Recall:",
    recall_score(y_test, y_pred)
)

print(
    "F1:",
    f1_score(y_test, y_pred)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_test, y_prob)
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 113. Complete SGDClassifier Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        random_state=42
    ))
])

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 114. Complete PassiveAggressive Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", PassiveAggressiveClassifier(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 115. Complete Perceptron Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", Perceptron(
        max_iter=1000,
        random_state=42
    ))
])

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 116. Complete RidgeClassifier Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RidgeClassifier(
        alpha=1.0
    ))
])

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 117. Complete RidgeClassifierCV Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RidgeClassifierCV(
        alphas=[0.01, 0.1, 1, 10, 100]
    ))
])

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

# 118. Model Comparison

| Model                       | Main Idea                            | Probability                  | Online Learning | Regularization                           |
| --------------------------- | ------------------------------------ | ---------------------------- | --------------- | ---------------------------------------- |
| LogisticRegression          | Logistic / log-loss classifier       | Yes                          | No              | L1/L2/ElasticNet depending configuration |
| SGDClassifier               | SGD-based linear classifier          | Sometimes, depending on loss | Yes             | L1/L2/ElasticNet                         |
| PassiveAggressiveClassifier | Passive-aggressive online classifier | No                           | Yes             | Controlled by C                          |
| Perceptron                  | Classic perceptron                   | No                           | Yes             | Optional regularization                  |
| RidgeClassifier             | Ridge / L2 classifier                | No                           | No              | L2                                       |
| RidgeClassifierCV           | Ridge + CV alpha selection           | No                           | No              | L2                                       |

---

# 119. Model Selection

Use-case mapping:

```text
Need probability prediction
→ LogisticRegression

Large dataset
→ SGDClassifier

Online / streaming learning
→ SGDClassifier
→ PassiveAggressiveClassifier
→ Perceptron

Simple classic linear classifier
→ Perceptron

L2-regularized linear classification
→ RidgeClassifier

Need automatic alpha selection with CV
→ RidgeClassifierCV

Text / sparse high-dimensional data
→ LogisticRegression
→ SGDClassifier
→ PassiveAggressiveClassifier
→ RidgeClassifier
```

This is a **use-case mapping**, not a universal ranking.

---

# 120. LogisticRegression vs SGDClassifier

```text
LogisticRegression
↓
Logistic loss
↓
Solver-based optimization
↓
Probability output
```

```text
SGDClassifier
↓
Stochastic Gradient Descent
↓
Multiple loss functions
↓
Online learning support
```

Main difference:

```text
Optimization strategy
```

---

# 121. LogisticRegression vs Perceptron

```text
LogisticRegression
→ Probability-based classifier
→ Log loss
→ Regularization
→ More suitable when probability estimates are needed

Perceptron
→ Classic mistake-driven linear classifier
→ Simple
→ Online learning
→ No probability output by default
```

---

# 122. LogisticRegression vs RidgeClassifier

```text
LogisticRegression
→ Log loss
→ Probability output
→ Classification likelihood-based objective

RidgeClassifier
→ Ridge / least-squares style objective
→ L2 regularization
→ Decision scores
```

---

# 123. SGDClassifier vs PassiveAggressiveClassifier

```text
SGDClassifier
→ Gradient-based optimization
→ Selected loss function
→ General-purpose SGD classifier

PassiveAggressiveClassifier
→ Aggressive update when prediction violates
  desired loss/margin condition
→ Online learning
```

---

# 124. Perceptron vs RidgeClassifier

```text
Perceptron
→ Perceptron update rule
→ Online learning
→ Simple linear classifier

RidgeClassifier
→ L2 regularized least-squares objective
→ Stable linear decision model
```

---

# 125. RidgeClassifier vs RidgeClassifierCV

```text
RidgeClassifier
→ alpha manually selected

RidgeClassifierCV
→ alpha selected using cross-validation
```

Example:

```text
RidgeClassifier
alpha = 1

RidgeClassifierCV
tries:
0.01
0.1
1
10
100
```

and selects a suitable value according to the CV procedure.

---

# 126. Classification Project Flow

Complete classification workflow:

```text
Dataset
   ↓
Understand Dataset
   ↓
Data Cleaning
   ↓
Handle Missing Values
   ↓
EDA
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Stratification
   ↓
Preprocessing
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Prediction
   ↓
Evaluation
   ↓
Cross Validation
   ↓
Hyperparameter Tuning
   ↓
Model Comparison
   ↓
Final Model
   ↓
Save Model
   ↓
Deploy
```

---

# 127. Classification Evaluation Flow

```text
Prediction
   ↓
Confusion Matrix
   ↓
Accuracy
   ↓
Precision
   ↓
Recall
   ↓
F1 Score
   ↓
ROC-AUC / PR-AUC
   ↓
Error Analysis
```

For imbalanced datasets:

```text
Do not rely only on Accuracy.
```

---

# 128. Model Saving

Using Joblib:

```python
import joblib

joblib.dump(
    model,
    "classification_model.pkl"
)
```

Load:

```python
model = joblib.load(
    "classification_model.pkl"
)
```

If the model is a Pipeline, save the entire Pipeline so preprocessing and prediction remain together.

---

# 129. Model Deployment

Typical deployment flow:

```text
User Input
    ↓
Preprocessing
    ↓
Trained Model
    ↓
Prediction
    ↓
Response
```

Example:

```text
User:
Age = 35
Salary = 60000
Credit Score = 720

        ↓

Classification Model

        ↓

Approved
```

For probability-based models:

```text
Prediction = Approved

Probability = 0.87
```

---

# 130. Common Classification Problems

Examples:

```text
Spam Detection
Fraud Detection
Disease Detection
Sentiment Analysis
Loan Approval
Customer Churn
Credit Risk
Image Classification
Document Classification
Face Recognition
```

---

# 131. Interview Questions — Beginner

1. What is Classification?

2. What is the difference between Regression and Classification?

3. What is Binary Classification?

4. What is Multiclass Classification?

5. What is Multilabel Classification?

6. What is a Class Label?

7. What is a Feature?

8. What is a Target?

9. What is a Decision Boundary?

10. What is a Linear Classifier?

11. What is Logistic Regression?

12. Why is Logistic Regression used for classification?

13. What is the Sigmoid Function?

14. What is Probability in Logistic Regression?

15. What is a Confusion Matrix?

16. What is Accuracy?

17. What is Precision?

18. What is Recall?

19. What is F1 Score?

20. What are TP, TN, FP, and FN?

---

# 132. Interview Questions — Intermediate

21. What is SGDClassifier?

22. What is Stochastic Gradient Descent?

23. What is PassiveAggressiveClassifier?

24. What is Perceptron?

25. What is RidgeClassifier?

26. What is RidgeClassifierCV?

27. Difference between RidgeClassifier and RidgeClassifierCV?

28. Difference between LogisticRegression and SGDClassifier?

29. Difference between SGDClassifier and Perceptron?

30. Difference between SGDClassifier and PassiveAggressiveClassifier?

31. Difference between LogisticRegression and RidgeClassifier?

32. What is `predict_proba()`?

33. What is `decision_function()`?

34. What is the difference between probability and decision score?

35. What is class imbalance?

36. How do you handle class imbalance?

37. What is `class_weight="balanced"`?

38. Why is `stratify=y` used?

39. What is Cross Validation?

40. What is StratifiedKFold?

41. What is Feature Scaling?

42. Why is scaling important for SGDClassifier?

43. What is Regularization?

44. What is L1 Regularization?

45. What is L2 Regularization?

46. What is Hyperparameter Tuning?

47. What is GridSearchCV?

48. What is RandomizedSearchCV?

---

# 133. Interview Questions — Advanced

49. Explain Logistic Regression mathematically.

50. What is the Sigmoid Function?

51. What is Log Loss?

52. What is Cross-Entropy Loss?

53. What are Odds and Log-Odds?

54. What is Odds Ratio?

55. How do you interpret a Logistic Regression coefficient?

56. How is a Logistic Regression decision boundary obtained?

57. What is the difference between a decision score and probability?

58. What is ROC Curve?

59. What is ROC-AUC?

60. What is Precision-Recall Curve?

61. When is PR-AUC more informative than ROC-AUC?

62. What is threshold tuning?

63. Why can accuracy be misleading?

64. What is calibration?

65. What is `CalibratedClassifierCV`?

66. What is Online Learning?

67. What is `partial_fit()`?

68. What is Data Leakage?

69. Why should preprocessing be inside a Pipeline?

70. What is Multicollinearity?

71. How does regularization help classification?

72. What is the difference between L1 and L2 regularization?

73. How would you classify text using Logistic Regression?

74. Why are linear classifiers effective for TF-IDF text features?

75. How do you select a classification model?

76. How do you evaluate an imbalanced classifier?

77. How do you tune the classification threshold?

78. How do you deploy a classification model?

---

# 134. Quick Revision Table

| Topic                       | Remember                                 |
| --------------------------- | ---------------------------------------- |
| Classification              | Predict discrete classes                 |
| Binary Classification       | Two classes                              |
| Multiclass                  | More than two classes                    |
| Multilabel                  | Multiple labels per sample               |
| LogisticRegression          | Log-loss based linear classifier         |
| Sigmoid                     | Converts binary score to probability     |
| SGDClassifier               | SGD-based linear classifier              |
| PassiveAggressiveClassifier | Online passive-aggressive learning       |
| Perceptron                  | Classic mistake-driven linear classifier |
| RidgeClassifier             | L2-regularized linear classifier         |
| RidgeClassifierCV           | RidgeClassifier + CV alpha selection     |
| Accuracy                    | Overall correct predictions              |
| Precision                   | Correctness of predicted positives       |
| Recall                      | Detection of actual positives            |
| F1                          | Harmonic mean of precision and recall    |
| TP                          | Actual positive, predicted positive      |
| TN                          | Actual negative, predicted negative      |
| FP                          | Actual negative, predicted positive      |
| FN                          | Actual positive, predicted negative      |
| ROC-AUC                     | Ranking performance across thresholds    |
| PR-AUC                      | Precision-recall performance             |
| Log Loss                    | Probability quality                      |
| Decision Function           | Model decision score                     |
| `predict_proba()`           | Probability estimates                    |
| `class_weight`              | Handles class imbalance                  |
| StandardScaler              | Feature standardization                  |
| Pipeline                    | Preprocessing + model                    |
| Cross Validation            | Multiple validation splits               |
| StratifiedKFold             | Preserves class proportions              |
| L1                          | Can make coefficients zero               |
| L2                          | Shrinks coefficients                     |
| Online Learning             | Learn incrementally                      |
| `partial_fit()`             | Incremental training                     |
| GridSearchCV                | Exhaustive specified parameter search    |
| RandomizedSearchCV          | Random parameter search                  |
| Calibration                 | Makes probabilities better calibrated    |

---

# 135. Final Short Summary

Classification:

```text
Predict Discrete Classes
```

Main Models:

```text
LogisticRegression
→ Probability-based linear classification

SGDClassifier
→ Stochastic Gradient Descent

PassiveAggressiveClassifier
→ Online passive-aggressive learning

Perceptron
→ Classic linear classifier

RidgeClassifier
→ L2-regularized classification

RidgeClassifierCV
→ Ridge classification + cross-validation
```

Important Mathematical Concepts:

```text
Linear Score:

z = wᵀx + b
```

Logistic Regression:

```text
p = 1 / (1 + e⁻ᶻ)
```

Log-Odds:

```text
log(p / (1-p))
```

Decision Boundary:

```text
wᵀx + b = 0
```

Important Metrics:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
PR-AUC
Log Loss
Balanced Accuracy
```

Confusion Matrix:

```text
              Predicted

              0       1

Actual 0      TN      FP

Actual 1      FN      TP
```

Regularization:

```text
L1
→ Can perform feature selection

L2
→ Shrinks coefficients

ElasticNet
→ L1 + L2
```

Scaling:

```text
StandardScaler
MinMaxScaler
```

Important for:

```text
SGDClassifier
PassiveAggressiveClassifier
Perceptron
```

and generally useful for regularized linear classifiers.

Cross Validation:

```text
K-Fold
StratifiedKFold
```

Hyperparameter Tuning:

```text
GridSearchCV
RandomizedSearchCV
```

Online Learning:

```text
SGDClassifier
PassiveAggressiveClassifier
Perceptron
```

Incremental training:

```python
model.partial_fit(...)
```

Pipeline:

```text
Preprocessing
      ↓
Classifier
```

Complete Classification Flow:

```text
Dataset
   ↓
EDA
   ↓
Cleaning
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Stratification
   ↓
Preprocessing
   ↓
Scaling
   ↓
Model Training
   ↓
Prediction
   ↓
Confusion Matrix
   ↓
Precision / Recall / F1
   ↓
ROC-AUC / PR-AUC
   ↓
Cross Validation
   ↓
Hyperparameter Tuning
   ↓
Threshold Tuning
   ↓
Final Model
   ↓
Save
   ↓
Deploy
```

## Final Model Selection Cheat Sheet

```text
Need probabilities
        ↓
LogisticRegression

Large-scale / SGD
        ↓
SGDClassifier

Online / Streaming
        ↓
SGDClassifier
PassiveAggressiveClassifier
Perceptron

Simple classic linear classifier
        ↓
Perceptron

L2 regularization
        ↓
RidgeClassifier

Automatic alpha selection
        ↓
RidgeClassifierCV

Text / Sparse Features
        ↓
LogisticRegression
SGDClassifier
PassiveAggressiveClassifier
RidgeClassifier
```
