🔁 **Stage 6: Feature Transformation**

---

### 🎯 Goal:

Transform numerical and categorical features to improve model accuracy and handle skewed distributions, non-linearity, and scaling issues.

---

## 📚 Key Concepts:

1. Scaling and Normalization
2. Power Transformation (Log, Box-Cox)
3. Binning (Discretization)
4. Interaction Features
5. Polynomial Features
6. Date and Time Features

---

## 1. ⚖️ Scaling and Normalization

### a. **Standardization (Z-score normalization)**

- Centers the mean at 0 and scales to unit variance.
- Formula: \(z = \frac{x - \mu}{\sigma}\), where \(\mu\) is the mean and \(\sigma\) is the standard deviation.
- Keeps outliers and negative values.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```

### b. **Min-Max Scaling (Normalization)**

- Transforms values between 0 and 1.
- Formula: \(x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}\)
- Compresses all features to a common scale but is sensitive to outliers.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```

### 🔍 Standardization vs Normalization:

| Aspect                | Standardization                                                                  | Normalization                                               |
| --------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Scale                 | Mean = 0, Std = 1                                                                | [0, 1] or [-1, 1]                                           |
| Sensitive to outliers | No                                                                               | Yes                                                         |
| Preserves outliers    | Yes                                                                              | No (compresses)                                             |
| Use case              | Algorithms assuming Gaussian (e.g., Linear Regression, Logistic Regression, SVM) | Algorithms relying on distance (e.g., KNN, Neural Networks) |

---

## 2. 🔋 Power Transformation

### a. **Log Transformation**

- Reduces skewness, handles wide distributions.

```python
df['Fare_log'] = np.log1p(df['Fare'])  # log(1 + x)
```

### b. **Box-Cox / Yeo-Johnson**

- More flexible transformations.

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df[['Fare']] = pt.fit_transform(df[['Fare']])
```

---

## 3. 🧊 Binning (Discretization)

Convert continuous variable into bins (e.g., age groups).

```python
df['Age_bin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
```

Use case: Decision tree models, or simplifying interpretation.

---

## 4. 🧩 Interaction Features

Combine two or more features to create new ones.

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
```

Use case: Logistic regression, tree models may benefit from implicit groupings.

---

## 5. 🧮 Polynomial Features

Generate higher-degree features to model non-linear relationships.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Age', 'Fare']])
```

⚠️ Increases dimensionality – use with care.

---

## 6. 📆 Date and Time Features

Extract insights from datetime columns.

```python
df['Hour'] = df['Timestamp'].dt.hour
df['Weekday'] = df['Timestamp'].dt.weekday
df['Is_Weekend'] = df['Weekday'].isin([5,6]).astype(int)
```

Use case: Time series, behavioral models, traffic prediction, etc.

---

## 📘 Real-World Example: Titanic

1. **Log Transform Fare**

```python
df['Fare_log'] = np.log1p(df['Fare'])
```

2. **Create Family Size**

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
```

3. **Binning Age**

```python
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
```

---

## ✅ Outcome of Stage 6:

By the end of this stage, your dataset will:

- Have normalized or standardized features
- Handle skewness and non-linear relationships
- Include engineered features that may boost model performance

---

Next Step: **Stage 7 - Feature Selection**

