# 🔁 Stage 4: Statistical Thinking for Machine Learning

In this stage, we move beyond simple averages and predictions to a deeper understanding of how variables relate and interact in a dataset. We'll address concepts that affect **data quality**, **feature selection**, and **model performance** — crucial for building effective machine learning pipelines.

We’ll explore:

- Correlation vs. Causation (Pearson, Spearman)
- Covariance
- Feature relationships (Multicollinearity, Outliers)
- Missing data handling (Mean/Median, KNN, Regression, MICE)
- Regression techniques for prediction

---

## 1. Correlation vs Causation

### 🔗 Correlation

Correlation measures whether two variables **move together**.

- **Pearson Correlation**: Measures **linear relationships** (e.g., hours studied vs. test scores).
- **Spearman Correlation**: Measures **rank-order relationships** (useful for non-linear trends or ordinal data).

> 🔍 Real-World Example:
>
> - **Pearson**: Height vs. Weight — taller people tend to weigh more (linear relationship).
> - **Spearman**: Class rank vs. stress level — rank goes down, stress goes up, even if not linearly.

### ⚠️ Causation

Just because A and B change together **doesn’t mean A causes B**.

- Always ask: Could both be caused by something else?

> 🧠 Example:
>
> - Ice cream sales and drowning deaths both increase in summer.
> - They correlate, but **temperature (confounder)** causes both.

### 🧰 Code Example (Pearson vs Spearman)

```python
import pandas as pd

x = pd.Series([1, 2, 3, 4, 5])
y = pd.Series([2, 4, 6, 8, 10])
z = pd.Series([10, 9, 7, 5, 1])

print("Pearson:", x.corr(y))       # Linear
print("Spearman:", x.corr(z, method='spearman'))  # Rank-based
```

---

## 2. Covariance

### ✅ Definition

Covariance tells you whether two variables **increase or decrease together**, but it does **not standardize the result**, so it's harder to interpret than correlation.

- **Positive covariance** → variables move in the same direction
- **Negative covariance** → move in opposite directions
- **Zero covariance** → no relationship

> 🌡️ Example: As temperature rises, ice cream sales increase → **positive covariance**

### 📌 Code Example

```python
import numpy as np

x = [2, 4, 6, 8, 10]
y = [1, 3, 5, 7, 9]
cov_matrix = np.cov(x, y)
print("Covariance matrix:\n", cov_matrix)
```

Use **correlation** when you want a standardized measure, and **covariance** when raw scale is informative (e.g., portfolio risk).

---

## 3. Feature Relationships

### 🔁 Multicollinearity

Multicollinearity occurs when **two or more features are highly correlated**.

- It can confuse regression models and inflate error.
- Detected using **VIF (Variance Inflation Factor)**:
  - VIF > 5 or 10 → multicollinearity concern

> 🏡 Example:
>
> - House size (sqft) and number of rooms are highly correlated.
> - One might be removed to reduce redundancy.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import numpy as np

X = np.array([[1000, 5], [1500, 7], [2000, 9], [2500, 11]])
X = add_constant(X)
print("VIFs:", [variance_inflation_factor(X, i) for i in range(X.shape[1])])
```

### ⚠️ Outlier Detection

Outliers can **skew your model** and reduce generalizability.

- **Z-score method**: flag points far from the mean
- **IQR method**: detect points outside [Q1 – 1.5×IQR, Q3 + 1.5×IQR]

> 🧪 Example:
>
> - Detect students with unusually high/low test scores

```python
from scipy.stats import zscore
import numpy as np

data = np.array([50, 52, 55, 58, 95])
z_scores = zscore(data)
outliers = np.where(np.abs(z_scores) > 2)
print("Outliers:", data[outliers])
```

---

## 4. Missing Data Techniques

### 🔧 Mean/Median Imputation

Simple, fast way to fill missing values:

- Use **mean** for symmetric data
- Use **median** for skewed data

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median")
data = [[1], [2], [np.nan], [4]]
print("Imputed:", imp.fit_transform(data))
```

### 🤝 KNN, Regression, MICE

These techniques use **relationships between features** to fill in missing data:

#### 🧠 KNN Imputation

- Finds the most similar rows (neighbors)
- Fills missing values based on their values

```python
from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors=2)
print(knn.fit_transform([[1, 2], [3, np.nan], [4, 5]]))
```

#### 📈 Regression Imputation

- Predict missing value using regression on other variables

> Example: Predict height from weight, age, and gender

#### 🔄 MICE (Multiple Imputation by Chained Equations)

- More advanced: performs multiple rounds of imputation using statistical models.
- Available via `fancyimpute` or `statsmodels.imputation`.

---

## 5. Regression (Simple & Multiple)

### ✅ Simple Linear Regression

Regression quantifies the **relationship** between an independent variable (X) and a dependent variable (Y), and allows **prediction**.

- Equation: **Y = a + bX + ε**

> 📈 Example: Predict house price from square footage

```python
from sklearn.linear_model import LinearRegression

X = np.array([1000, 1500, 2000, 2500]).reshape(-1, 1)
y = np.array([200000, 250000, 300000, 350000])

model = LinearRegression()
model.fit(X, y)
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
```

### 🔢 Multiple Regression

Use more than one variable to make better predictions

> Example: Predict house price from size, bedrooms, and location

```python
X_multi = np.array([[1000, 3], [1500, 4], [2000, 5]])
y_price = [200000, 300000, 400000]

model_multi = LinearRegression()
model_multi.fit(X_multi, y_price)
print("Coefficients:", model_multi.coef_)
```

---

## ✅ Summary Table

| Concept                | Goal                             | Real-World Application                   |
| ---------------------- | -------------------------------- | ---------------------------------------- |
| Correlation            | Measure variable relationships   | Study time ↔️ Exam score                 |
| Causation              | Show influence/cause             | Ad budget → Sales (after testing)        |
| Covariance             | Track direction of joint change  | Temp ↑ → Ice cream sales ↑               |
| Multicollinearity      | Remove redundant features        | Home size & #rooms in pricing model      |
| Outlier Detection      | Handle unusual values            | Salary, score spikes                     |
| Mean/Median Imputation | Fast missing value fix           | Fill gaps in survey/CSV data             |
| KNN/Regression/MICE    | Smart imputations using patterns | Medical records, housing data completion |
| Simple Regression      | Predict using one variable       | Price from house size                    |
| Multiple Regression    | Predict using many features      | Price from size, rooms, location         |

---

In **Stage 5**, we’ll dive deeper into **Feature Engineering & Selection** — crafting better inputs to boost model power 🚀

