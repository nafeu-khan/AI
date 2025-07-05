📉 **Stage 2: Handling Missing Data**

---

### 🎯 Goal:

Handle missing values efficiently using different strategies depending on the data type, domain knowledge, and impact on the model.

---

## 🔍 Common Techniques:

### 1. **Remove Rows/Columns with Missing Data**

- When: If only a small % of rows or a column with too many missing values.

```python
# Drop rows with any missing values
df.dropna(inplace=True)

# Drop columns with >50% missing
df.dropna(thresh=len(df)*0.5, axis=1, inplace=True)
```

### 2. **Simple Imputation**

- Use **mean**, **median**, or **mode** based on feature type.

```python
from sklearn.impute import SimpleImputer

# Median for numerical features
imp_median = SimpleImputer(strategy='median')
df['Age'] = imp_median.fit_transform(df[['Age']])

# Mode for categorical
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

### 3. **Group-based Imputation**

- Impute values based on groups (e.g., median age per Pclass)

```python
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
```

### 4. **Model-based Imputation**

- Predict missing values using another ML model (e.g., RandomForestRegressor)

### 5. **KNN Imputation**

```python
from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors=5)
df_imputed = knn.fit_transform(df.select_dtypes(include=['float64', 'int64']))
```

### 6. **MICE (Multivariate Imputation by Chained Equations)**

- Iteratively predicts each feature based on others

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

mice = IterativeImputer()
df_imputed = mice.fit_transform(df.select_dtypes(include=['float64', 'int64']))
```

---

## 📘 Real-World Example: Titanic

- Problem: `Age` and `Cabin` have many missing values.
- Solution:
  - Drop `Cabin` (too sparse)
  - Impute `Age` using median by `Pclass`
  - Impute `Embarked` using mode

---

## ⚠️ Tips:

- Visualize missingness using `missingno` or `sns.heatmap(df.isnull())`
- Never impute test data using test statistics — always fit on training data.
- Evaluate if imputation affects performance (before/after comparison).

---

## ✅ Outcome of Stage 2:

You should now have:

- No significant missing values
- Imputed values backed by logic
- Cleaner dataset ready for transformation

---

Next Step: **Stage 3 - Cleaning & Fixing Data**

