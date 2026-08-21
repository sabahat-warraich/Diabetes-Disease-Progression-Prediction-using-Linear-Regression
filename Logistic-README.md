# Logistic Regression – Insurance Purchase Prediction

## Overview

This project demonstrates **Logistic Regression** using Python and scikit-learn.

The goal is to predict whether a person will **purchase insurance (`1`) or not purchase insurance (`0`) based on their age**.

This project also demonstrates how Logistic Regression produces **probabilities**, how predictions are made from those probabilities, and how to calculate **Log Loss** manually.

---

## What is Logistic Regression?

**Logistic Regression is a supervised machine learning algorithm used mainly for classification problems.**

Unlike Linear Regression, which predicts a continuous numerical value, Logistic Regression predicts the probability of an observation belonging to a particular class.

For example:

| Age | Purchased |
| --: | --------: |
|  22 |         0 |
|  25 |         0 |
|  35 |         1 |
|  42 |         1 |
|  50 |         1 |

Here:

* `0` → Did not purchase insurance
* `1` → Purchased insurance

The model learns the relationship between **Age** and **Purchased** and then predicts the probability of purchase for a new person.

---

## Why is it called Logistic Regression?

Although the name contains "Regression", Logistic Regression is primarily used for **classification**.

The model first calculates a linear combination:

```text
z = b₀ + b₁x
```

It then passes this value through the **Sigmoid Function**.

The sigmoid function converts any value into a probability between `0` and `1`.

```text
        1
P =  ---------
     1 + e⁻ᶻ
```

The output can be interpreted as the probability of belonging to class `1`.

For example:

```text
Probability = 0.85
```

means the model estimates an **85% probability** that the person belongs to class `1`.

---

## How Classification Works

The model converts the probability into a class using a threshold.

A common threshold is `0.5`.

```text
Probability >= 0.5  →  Class 1
Probability <  0.5  →  Class 0
```

For example:

```text
Age = 25
Probability = 0.20
Prediction = 0

Age = 45
Probability = 0.82
Prediction = 1
```

The threshold can be changed depending on the requirements of the problem.

---

## Dataset

The project uses an `insurance.csv` dataset.

The important columns are:

### Age

The age of the person.

### Purchased

The target variable.

```text
0 = Not Purchased
1 = Purchased
```

In this project:

```python
X = df[['Age']]
y = df['Purchased']
```

`Age` is the input feature, while `Purchased` is the target variable.

---

## Project Workflow

The project follows these machine learning steps:

```text
Load Dataset
     ↓
Visualize Data
     ↓
Select Features and Target
     ↓
Split Data
     ↓
Train Logistic Regression
     ↓
Make Predictions
     ↓
Calculate Probabilities
     ↓
Calculate Log Loss
     ↓
Visualize Results
```

---

## 1. Import Libraries

The project uses:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

And scikit-learn for splitting the data and training the model:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```

---

## 2. Load the Dataset

```python
df = pd.read_csv("insurance.csv")
```

This loads the insurance dataset into a pandas DataFrame.

---

## 3. Visualize the Dataset

The relationship between age and insurance purchase is visualized using a scatter plot:

```python
plt.scatter(
    df.Age,
    df.Purchased,
    marker='+',
    color='red'
)
```

This allows us to visually inspect whether age has a relationship with purchasing insurance.

---

## 4. Select Features and Target

```python
X = df[['Age']]
y = df['Purchased']
```

Here:

```text
X = Input Feature
y = Target
```

So the model uses:

```text
Age → Purchased
```

---

## 5. Train/Test Split

The dataset is divided into training and testing data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

`test_size=0.2` means that:

* 80% of the data is used for training
* 20% is used for testing

The model learns from the training data and is evaluated using previously unseen test data.

---

## 6. Train Logistic Regression

```python
model = LogisticRegression()

model.fit(X_train, y_train)
```

During training, the model learns the parameters needed to estimate the probability of purchasing insurance based on age.

Conceptually, it learns:

```text
Age → Linear Score → Sigmoid → Probability
```

---

## 7. Make Predictions

The `predict()` method returns the final class:

```python
predictions = model.predict(X_test)
```

The result will contain values such as:

```text
[0 1 1 0 1 ...]
```

These are the model's final classification decisions.

---

## 8. Predict Probabilities

Logistic Regression can also return probabilities:

```python
y_pred_proba = model.predict_proba(X_test)[:, 1]
```

For binary classification, `predict_proba()` returns probabilities for both classes:

```text
Class 0 Probability    Class 1 Probability
       0.80                   0.20
       0.25                   0.75
       0.10                   0.90
```

The following:

```python
[:, 1]
```

selects the probability of **class 1**.

For example:

```text
[0.20, 0.75, 0.90, 0.35]
```

means the model predicts:

```text
20% probability of purchase
75% probability of purchase
90% probability of purchase
35% probability of purchase
```

---

# Log Loss

Log Loss is a metric used to evaluate the quality of probabilistic classification predictions.

It doesn't only look at whether the prediction is correct or incorrect. It also considers **how confident the model was**.

The binary Log Loss formula is:

```text
                 1
Log Loss = -  -------- Σ [ y log(p) + (1-y) log(1-p) ]
                 n
```

Where:

* `y` = actual class
* `p` = predicted probability of class `1`
* `n` = number of observations

### Example

Suppose the actual value is:

```text
y = 1
```

Prediction A:

```text
p = 0.90
```

This is a good prediction because the model was highly confident in the correct class.

Prediction B:

```text
p = 0.10
```

This is a very bad prediction because the model was highly confident in the wrong class.

Therefore:

> **Lower Log Loss is better.**

---

## Manual Log Loss Implementation

This project calculates Log Loss manually:

```python
def log_loss(y_true, y_pred):

    n = len(y_true)

    loss = -(1/n) * np.sum(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss
```

Then:

```python
loss = log_loss(
    y_test.values,
    y_pred_proba
)

print("Log Loss:", loss)
```

This helps demonstrate what happens internally instead of simply using a pre-built metric.

---

## Visualization

The project can visualize the Logistic Regression model using:

### Actual Data

```text
Age → Purchased
```

This shows the original observations.

### Logistic Regression Probability Curve

The model's predicted probability can be plotted against age.

Conceptually:

```text
Probability
1.0 |                         ______
    |                     ___/
    |                  __/
0.5 |---------------__/
    |            ___/
    |         __/
0.0 |________/
    +---------------------------- Age
```

This S-shaped curve is produced by the **Sigmoid Function**.

It shows how the probability of purchasing insurance changes as age increases.

---

## Important Difference: Linear vs Logistic Regression

### Linear Regression

Used for predicting continuous values.

Example:

```text
House Size → House Price
```

Output:

```text
250000
350000
450000
```

### Logistic Regression

Used mainly for classification.

Example:

```text
Age → Insurance Purchase
```

Output:

```text
0 or 1
```

or a probability:

```text
0.15
0.72
0.94
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Logistic Regression

---

## Key Concepts Learned

This project demonstrates:

* Supervised Learning
* Binary Classification
* Logistic Regression
* Sigmoid Function
* Probability Prediction
* Classification Threshold
* Train/Test Split
* Model Prediction
* Log Loss
* Data Visualization

---

## How to Run

Clone the repository:

```bash
git clone <your-repository-url>
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Make sure `insurance.csv` is in the project directory.

Then run the Python script or Jupyter Notebook.

---

## Conclusion

This project provides a simple introduction to **Logistic Regression** using an insurance purchase prediction problem.

The main idea is:

```text
Input Feature
     ↓
Linear Equation
     ↓
Sigmoid Function
     ↓
Probability
     ↓
Classification
```

For this project:

```text
Age
 ↓
Logistic Regression
 ↓
Probability of Purchase
 ↓
0 or 1
```

The project also demonstrates why evaluating only the final `0/1` prediction is not always enough. **Probability predictions and Log Loss provide additional information about how confident and accurate the model is.**
