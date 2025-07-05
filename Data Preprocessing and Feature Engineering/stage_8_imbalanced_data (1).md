⚖️ **Stage 8: Handling Imbalanced Datasets**

---

### 🎯 Goal:

Effectively address skewed class distributions in classification problems to prevent biased model predictions.

---

## 🧠 Why It Matters:

In real-world datasets, especially in domains like fraud detection, medical diagnosis, or churn prediction, one class may vastly outnumber the other. A model trained on such data might predict only the majority class and achieve deceptively high accuracy. This can be dangerous — especially when the minority class represents critical cases (like fraud or disease).

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

This extreme imbalance may cause a naive model to predict only class 0 and still score 98% accuracy — but completely miss all important minority cases.

---

## 📚 Common Techniques

### 1. **Resampling**

Change the dataset distribution to balance class ratios.

#### a. **Random Oversampling**

Duplicates examples from the minority class. Pros: Easy, retains all majority class examples Cons: Overfitting risk due to repeated samples

```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)
```

#### b. **SMOTE (Synthetic Minority Over-sampling Technique)**

Creates synthetic samples based on feature-space similarities. Pros: More diverse than duplication Cons: Can create noise near class boundaries

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
```

#### c. **Random Undersampling**

Removes samples from the majority class. Pros: Fast, reduces training time Cons: Risk of losing important data

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X, y)
```

---

### 2. **Class Weighting**

Inform the model to penalize mistakes on the minority class more heavily. Ideal when you want to preserve the original data distribution.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
```

You can also manually assign weights based on inverse class frequency.

---

### 3. **Evaluation Metrics for Imbalanced Data**

Relying on **accuracy** is misleading. Use the following instead:

| Metric               | Description                                         |
| -------------------- | --------------------------------------------------- |
| Precision            | True Positives / (True Positives + False Positives) |
| Recall (Sensitivity) | True Positives / (True Positives + False Negatives) |
| F1 Score             | Harmonic mean of Precision and Recall               |
| ROC-AUC              | Area under ROC curve                                |
| PR-AUC               | Area under Precision-Recall curve                   |

```python
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
```

---

## 📘 Real-World Example: Credit Card Fraud Dataset

- Imbalance: 0.17% fraud cases
- Goal: Accurately detect rare fraudulent transactions
- Strategy: Use SMOTE to oversample + RandomForest with class weighting

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_res, y_res)
```

---

## 🧪 Best Practices:

- Split into **train/test BEFORE** resampling to prevent data leakage
- Use **cross-validation** with stratified folds
- Try **ensemble methods** like EasyEnsemble or BalancedBaggingClassifier
- Plot both **ROC and PR curves** for performance comparison
- Monitor **confusion matrix** closely

---

## ✅ Outcome of Stage 8:

- You have rebalanced your dataset using appropriate strategy (resampling or class weighting)
- You used **recall**, **F1**, and **ROC-AUC** to fairly evaluate performance
- Your model now recognizes both majority and minority class instances better

---

Next Step: **Stage 9 - Noise Removal & Data Quality Checks**

