🧽 **Stage 9: Noise Removal & Data Quality Checks**

---

### 🎯 Goal:

Ensure your dataset is free from inconsistencies, anomalies, or erroneous records that can mislead the model during training and evaluation.

---

## 🧠 Why It Matters:

- Dirty data leads to misleading patterns
- Models may overfit to noisy or incorrect labels
- Quality preprocessing results in better performance and generalization

---

## 📚 Key Focus Areas:

1. Duplicate Records
2. Inconsistent Categories and Typos
3. Low Variance Features
4. Label Noise (Incorrect Labels)
5. Outlier Inspection

---

## 1. 🔁 Duplicate Records

- Repeated rows can bias models, especially tree-based and distance-based ones.

```python
# Check and drop duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)
```

---

## 2. ✏️ Fix Inconsistent Categories / Typos

- Inconsistent string formatting can lead to unintended splits.

```python
# Lowercase, strip, replace typos
df['gender'] = df['gender'].str.lower().str.strip()
df['gender'].replace({'femail': 'female'}, inplace=True)
```

- Use `.value_counts()` to spot typos or rare categories:

```python
df['gender'].value_counts()
```

---

## 3. 🧊 Remove Low Variance Features

- Features with no variability add noise but no value.

```python
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
X_var = sel.fit_transform(X)
```

---

## 4. ❗ Identify Label Noise

- Incorrect labels in supervised datasets can corrupt model learning
- Use:
  - Domain knowledge to verify data
  - Model disagreement methods
  - Cross-validation + predictions for anomaly spotting

### Example: Cross-validated Predictions

```python
from sklearn.model_selection import cross_val_predict
probs = cross_val_predict(model, X, y, cv=5, method='predict_proba')

# Identify records where model is very confident but label disagrees
suspect = (probs[:,1] > 0.95) & (y == 0)
df[suspect]
```

---

## 5. 📏 Outlier Detection (Final Sweep)

You may have already performed outlier removal (Stage 5), but another review post-cleaning is useful.

- Use IQR, Z-score, IsolationForest, DBSCAN depending on data type

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.01)
df['anomaly'] = iso.fit_predict(X)
df = df[df['anomaly'] == 1]
```

---

## 📘 Real-World Example: Retail Sales Dataset

- Removed duplicate rows and records with negative sales
- Fixed typos in `store_type` column ('a ', ' A', 'A', etc.)
- Removed constant column `currency` (always 'USD')

---

## 🧪 Best Practices:

- Validate data types (e.g., dates should be `datetime64`)
- Normalize text inputs
- Review features with 90%+ missing or identical values
- Use visual tools: `pandas_profiling`, `sweetviz`, `dtale`

---

## ✅ Outcome of Stage 9:

- Clean, consistent, and trustworthy dataset
- Reduced risk of misleading model behavior
- Ready for feature modeling and validation

---

Next Step: **Stage 10 - Preventing Data Leakage**

