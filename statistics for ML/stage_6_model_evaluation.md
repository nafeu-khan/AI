# 🧪 Stage 6: Model Evaluation & Validation

This stage focuses on assessing how well a machine learning model performs and whether its performance is trustworthy. Evaluation isn’t just about a score — it’s about **understanding strengths, weaknesses, and generalization** to unseen data.

---

## 1. Holdout Validation Techniques

### 🔹 Train/Test Split

- Divide dataset into **training** and **testing** sets (e.g., 80/20)
- Ensures model is evaluated on unseen data

📊 *Example*: Predict house prices using 80% of the data, test on the remaining 20%

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 🔁 K-Fold Cross Validation

- Split data into **k** equal folds (e.g., 5)
- Train/test model **k** times, each time using a different fold as the test set
- Reduces variance in performance estimation

📊 *Example*: Use 5-fold CV to evaluate a logistic regression model for credit scoring

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(), X, y, cv=5)
print(scores)
```

### ⏫ Stratified K-Fold

- Ensures each fold has **same class distribution**
- Especially useful for imbalanced classification tasks

📊 *Example*: Cancer detection dataset with 5% positive cases

---

## 2. Bias-Variance Tradeoff

- **Bias**: Error due to overly simplistic assumptions
- **Variance**: Error due to sensitivity to small fluctuations in data
- Goal: Find a **balance** — not too simple (high bias), not too complex (high variance)

📊 *Example*:

- High bias: Linear model on spiral data → underfitting
- High variance: Deep decision tree → overfitting training data

---

## 3. Overfitting & Underfitting

### Overfitting

- Model performs well on training data but poorly on unseen data
- Learns noise instead of signal

### Underfitting

- Model is too simple to capture underlying patterns

📊 *Example*:

- Overfit: Polynomial regression of degree 15
- Underfit: Predicting income using only age

---

## 4. Evaluation Metrics by Task Type

### Classification

- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

📊 *Example*: Email spam detection — prioritize **recall** to catch all spam

### Regression

- Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of Determination)

📊 *Example*: Predicting housing prices — use **MAE** to minimize over/underpricing

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

---

## 5. Learning Curves

- Show how performance changes with increasing training data
- Helps diagnose underfitting/overfitting

📊 *Example*: Classifier accuracy vs. # of training examples

---

## 6. Model Comparison & Selection

- Compare models using **cross-validated metrics**
- Use statistical tests (paired t-tests, Wilcoxon signed-rank) to compare multiple models

📊 *Example*: Compare SVM, Logistic Regression, and Random Forest on F1 score

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = [RandomForestClassifier(), SVC(probability=True)]
for model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print(model.__class__.__name__, scores.mean())
```

---

## ✅ Summary Table

| Concept                  | Purpose                       | Real-World Example                |
| ------------------------ | ----------------------------- | --------------------------------- |
| Train/Test Split         | Quick evaluation              | Basic ML experiment               |
| K-Fold CV                | Stable performance estimate   | Competition or small data         |
| Stratified K-Fold        | Fair split for classification | Imbalanced disease classification |
| Bias/Variance            | Understand model error type   | Model tuning                      |
| Overfitting/Underfitting | Assess generalization         | Training vs. test performance     |
| Accuracy, F1, AUC        | Classification performance    | Spam detection, fraud detection   |
| MAE, RMSE, R²            | Regression performance        | Price prediction, time estimation |
| Learning Curves          | Understand model improvement  | Data acquisition decision         |
| Model Comparison         | Choose best-performing model  | A/B model testing for deployment  |

---

Next: **Stage 7 – Probability & Bayesian Thinking** 🤔

