🔒 **Stage 10: Preventing Data Leakage**

---

### 🎯 Goal:

Ensure that your machine learning model does not inadvertently access data during training that would not be available in a real-world prediction scenario.

---

## 🧠 Why It Matters:

- Leakage leads to **unrealistically high performance** during training and validation
- Models trained with leakage often **fail in production** due to overestimated capability
- Leakage is subtle — even seasoned practitioners may overlook it

Data leakage gives your model access to information it shouldn't have — making the model look good on paper, but perform poorly on real-world tasks. Detecting and preventing leakage is one of the most critical skills in applied ML.

Even small leaks can dramatically boost model accuracy during validation but yield a useless model when deployed. Therefore, rigorously simulating the actual prediction environment is essential.

---

## 📚 Common Types of Data Leakage

### 1. **Train-Test Contamination**

Occurs when information from the test set influences the training process, often during resampling or preprocessing.

#### ❌ Wrong:

Apply SMOTE or transformations to full data before splitting:

```python
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
```

#### ✅ Correct:

Always split first:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
```

📌 **Note**: This applies to PCA, feature selection, imputing, scaling, etc. Fit transformations only on training data.

---

### 2. **Feature Leakage (Leaky Predictors)**

Features that are strongly correlated with the target — because they were created after the target event — can cause the model to cheat.

#### Common Leaky Columns:

- Status flags (e.g., `is_churned`, `is_fraud`) already embedded in training data
- Transaction summaries that include future data
- Flags added manually during data collection
- Engineered features containing label information (e.g., "ReturnStatus" to predict "Return")

#### Fix:

- Remove these features
- Use only information available **before or at** prediction time
- Ask domain experts to verify the timeline of feature availability

---

### 3. **Temporal Leakage**

Especially dangerous in time-based data (finance, health, customer behavior).

#### Example:

Predicting if a user will churn in April, but features include actions taken in May.

#### Fix:

- Sort data chronologically
- Use time-based validation (like forward chaining)

```python
from sklearn.model_selection import TimeSeriesSplit
TimeSeriesSplit(n_splits=5)
```

- Avoid using future information for prediction

🕒 Also ensure training/test splits maintain time order. Never shuffle time series data.

---

### 4. **Preprocessing Leakage**

Occurs when operations like scaling or imputing are done **before** the train-test split, which causes the model to learn distributional characteristics of the full dataset.

#### ❌ Incorrect:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Test set influenced
```

#### ✅ Correct:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### 🔐 Best Practice:

Use `Pipeline`:

```python
from sklearn.pipeline import Pipeline
Pipeline([
  ('scaler', StandardScaler()),
  ('classifier', LogisticRegression())
])
```

Encapsulating transformations avoids human error and ensures clean separation.

---

## 🧪 Tips to Prevent Leakage:

- Always simulate real-world prediction timelines
- Don’t use post-outcome features (e.g., hospital readmission to predict first-time illness)
- Use data versioning tools (e.g., DVC, MLflow) to log transformations
- Apply transformations only inside cross-validation folds if tuning
- Validate all feature generation logic with domain knowledge
- Avoid using aggregate historical data that includes future labels

🧭 Rule of thumb: if you wouldn’t have the value in production **at prediction time**, don’t use it for training.

---

## 📘 Real-World Example: Telco Churn Dataset

- **Leakage Problem**: `TotalCharges` column was used, but it was only known at the end of a customer's lifecycle.
- **Fix**:

```python
df['EstTotalCharges'] = df['MonthlyCharges'] * df['tenure']
df.drop(columns=['TotalCharges'], inplace=True)
```

This simulates a real-world scenario where `TotalCharges` must be estimated rather than known.

---

## 🚨 Common Red Flags of Leakage:

| Symptom                                         | Possible Cause                                |
| ----------------------------------------------- | --------------------------------------------- |
| Accuracy near 99%                               | Target leakage or duplicate target in feature |
| Sharp train/test score drop                     | Preprocessing leakage or temporal mismatch    |
| Feature importances show target-derived columns | Direct leak through included labels           |
| High correlation between input & target         | Embedded target info in a feature             |
| Identical distributions in test/train sets      | Data reshuffling before proper splitting      |

---

## ✅ Outcome of Stage 10:

- Your model only uses data that would be available in a production scenario
- Evaluation metrics are realistic and trustworthy
- Your pipeline is safe for deployment in live environments
- Leakage risks are minimized through robust feature logic and validation splits

---

Next Step: **Stage 11 - Final Feature Matrix Preparation & Model Training**

