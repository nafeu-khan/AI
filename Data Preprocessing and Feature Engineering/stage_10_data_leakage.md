🔒 **Stage 10: Preventing Data Leakage**

---

### 🎯 Goal:

Ensure that your machine learning model does not inadvertently access data during training that would not be available in a real-world prediction scenario.

---

## 🧠 Why It Matters:

- Leakage leads to **unrealistically high performance** during training and validation
- Models trained with leakage often **fail in production** due to overestimated capability
- Leakage is subtle — even seasoned practitioners may overlook it

---

## 📚 Common Types of Data Leakage

### 1. **Train-Test Contamination**

Occurs when information from the test set influences the training process.

#### ❌ Wrong:

Apply SMOTE on the entire dataset, then split:

```python
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
```

#### ✅ Correct:

Split first, then apply SMOTE only on training data:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
```

💡 **Why?** Oversampling creates synthetic data from neighbors — if test data is present, the synthetic data leaks info.

---

### 2. **Feature Leakage (Leaky Predictors)**

These are features that contain information about the target — either explicitly or implicitly.

#### Common Examples:

- Using `TotalSpent` to predict churn, but `TotalSpent` includes post-churn billing.
- A column like `Outcome` present in the dataset.
- Including `is_returned` while predicting `return`.

#### Fix:

- Drop post-event or target-derived features.
- Simulate "real-time" feature availability.

---

### 3. **Temporal Leakage**

Occurs when using data that comes from **after** the point of prediction — often in time series or event data.

#### Example:

Using a user's 6-month click summary to predict whether they’ll churn in month 3.

#### Fix:

- Split data chronologically using `TimeSeriesSplit` or custom logic
- Never allow future information into the past model

```python
from sklearn.model_selection import TimeSeriesSplit
splitter = TimeSeriesSplit(n_splits=5)
```

---

### 4. **Preprocessing Leakage**

This happens when scaling, imputing, or encoding is done **before** the train-test split, resulting in test data influencing the transformations.

#### ❌ Incorrect:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### ✅ Correct:

```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

🛡️ Best solution: use `Pipeline` or `ColumnTransformer` to encapsulate preprocessing:

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
```

---

## 🧪 Tips to Prevent Leakage:

- Always ask: "Would this feature be available at prediction time in production?"
- Use pipelines to isolate training logic
- Validate suspicious features with domain experts
- In time-based data, use date-aware splits and avoid look-ahead
- Remove target-related columns before training
- Log transformation steps and assumptions

---

## 📘 Real-World Example: Telco Churn Dataset

- Leakage: Included `TotalCharges`, which accumulates over time and isn’t available when predicting at sign-up
- Fix:

```python
df['EstTotalCharges'] = df['MonthlyCharges'] * df['tenure']
df.drop(columns=['TotalCharges'], inplace=True)
```

---

## 🚨 Common Red Flags of Leakage:

| Symptom                                         | Possible Cause                                |
| ----------------------------------------------- | --------------------------------------------- |
| Accuracy near 99%                               | Target leakage or duplicate target in feature |
| Sharp train/test score drop                     | Leakage during cross-validation               |
| Feature importances show target-derived columns | Leaky predictors                              |

---

## ✅ Outcome of Stage 10:

- Your model only uses data that would be available in a production scenario
- Evaluation metrics are realistic and trustworthy
- Your pipeline is safe for deployment in live environments

---

Next Step: **Stage 11 - Final Feature Matrix Preparation & Model Training**

