🔬 **Stage 5: Statistical Thinking for Machine Learning**

---

### 🎯 Goal:

Understand the underlying statistical relationships and assumptions in your data to make better modeling decisions and avoid misleading results.

---

## 📚 Topics Covered:

1. Correlation vs Causation
2. Feature Relationships
3. Outlier Detection
4. Missing Data Mechanisms

---

## 1. 🧠 Correlation vs. Causation

- **Correlation**: A statistical relationship between two variables.
- **Causation**: One variable actually causes the other to change.

### 📘 Example:

> Ice cream sales and drowning deaths are correlated (due to summer), but eating ice cream doesn't cause drowning.

### 💡 Tools:

```python
# Pearson (linear)
df.corr(method='pearson')

# Spearman (monotonic)
df.corr(method='spearman')
```

---

## 2. 🔗 Feature Relationships

### a. **Multicollinearity**

When two or more features are highly correlated with each other. Can cause instability in linear models.

### 🔍 Detection:

- Correlation Matrix
- **VIF** (Variance Inflation Factor)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[['Age', 'Fare', 'Pclass']]

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
```

### 🎯 Rule:

- VIF > 5 or 10 → Potential multicollinearity issue

### b. **Confounding Variables**

- Variables that influence both independent and dependent variables, distorting the true effect.

---

## 3. 🧪 Outlier Detection

### a. **Z-score Method**

```python
from scipy.stats import zscore
z_scores = zscore(df[['Age', 'Fare']])
outliers = (abs(z_scores) > 3)
```

### b. **IQR Method**

```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]
```

---

## 4. 🤕 Missing Data Mechanisms

Understanding the **reason** why data is missing helps decide the right imputation technique.

- **MCAR** (Missing Completely At Random): No pattern
- **MAR** (Missing At Random): Missingness depends on observed data
- **MNAR** (Missing Not At Random): Missingness depends on unobserved data

### 📘 Titanic Example:

- `Age` is MAR (related to Pclass)
- `Cabin` is MNAR (missing more in lower classes)

---

## ✅ Outcome of Stage 5:

By the end of this stage, you should be able to:

- Analyze correlations between features
- Detect and manage multicollinearity
- Understand causes of missing data and handle accordingly
- Remove or treat statistical outliers

---

Next Step: **Stage 6 - Feature Transformation**

