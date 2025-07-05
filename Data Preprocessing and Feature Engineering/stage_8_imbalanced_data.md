⚖️ **Stage 8: Handling Imbalanced Datasets**

---

### 🎯 Goal:

Effectively address skewed class distributions in classification problems to prevent biased model predictions.

---

## 🧠 Why It Matters:

In real-world datasets, especially in domains like fraud detection, medical diagnosis, or churn prediction, one class may vastly outnumber the other. A model trained on such data might predict only the majority class and achieve deceptively high accuracy.

---

## 📉 Detecting Imbalance:

```python
# Visualize class distribution
import seaborn as sns
sns.countplot(x='target', data=df)

# Check percentage
df['target'].value_counts(normalize=True)
```

### Example:

- Class 0: 98%
- Class 1 (rare): 2%

---

## 📚 Common Techniques

### 1. **Resampling**

#### a. **Random Oversampling**

Duplicate minority samples to balance class distribution.

```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)
```

#### b. **SMOTE (Synthetic Minority Over-sampling Technique)**

Generates synthetic examples rather than duplicating.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
```

#### c. **Random Undersampling**

Removes examples from majority class.

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X, y)
```

---

### 2. **Class Weighting**

Many models allow class weights to penalize the majority class.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
```

---

### 3. **Evaluation Metrics for Imbalanced Data**

Accuracy alone is misleading. Use:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall Curve

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## 📘 Real-World Example: Credit Card Fraud Dataset

- Imbalance: 0.17% fraud
- Apply: SMOTE + RandomForestClassifier with `class_weight='balanced'`

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_res, y_res)
```

---

## 🧪 Tips:

- Always split train/test **before** applying SMOTE or oversampling to avoid data leakage.
- Try **ensemble methods** (e.g., EasyEnsemble, BalancedBaggingClassifier)
- Monitor **ROC-AUC** and **PR curves** rather than just accuracy

---

## ✅ Outcome of Stage 8:

- Balanced training data using appropriate strategy
- Models evaluated with fair metrics
- Improved detection of minority class events

---

Next Step: **Stage 9 - Noise Removal & Data Quality Checks**

