🎯 **Stage 7: Feature Selection**

---

### 🎯 Goal:

Reduce dimensionality by selecting the most relevant features, improving model performance, generalization, and interpretability.

---

## 🧠 Why Feature Selection Matters:

- Removes noisy and irrelevant data
- Reduces overfitting risk
- Speeds up training time
- Enhances model explainability

---

## 🔍 When to Perform Feature Selection:

- Before final model training (after cleaning, encoding, transforming)
- Especially important for high-dimensional data (e.g., text, genomics)

---

## 📚 Key Techniques

### 1. **Filter Methods**

- Evaluate the relevance of features by statistical relationships with the target.
- Fast and independent of ML algorithms.

#### a. **Correlation Coefficient**

- Use Pearson for linear, Spearman for monotonic relationships.

```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

#### b. **Chi-Squared Test** (for categorical target)

```python
from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(score_func=chi2, k=5).fit_transform(X, y)
```

#### c. **ANOVA F-test** (for regression)

```python
from sklearn.feature_selection import f_classif
scores = f_classif(X, y)
```

---

### 2. **Wrapper Methods**

- Search through subsets of features by training models repeatedly.
- More accurate but computationally expensive.

#### a. **Recursive Feature Elimination (RFE)**

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)
selected = X.columns[rfe.support_]
```

#### b. **Forward/Backward Selection**

- Forward: Start with none, add features one by one.
- Backward: Start with all, remove least useful step-by-step.

---

### 3. **Embedded Methods**

- Feature selection occurs during model training.

#### a. **Lasso Regression (L1 Regularization)**

- Shrinks less important feature coefficients to zero.

```python
from sklearn.linear_model import LassoCV
model = LassoCV().fit(X, y)
selected = X.columns[model.coef_ != 0]
```

#### b. **Tree-Based Models (e.g., Random Forest)**

- Use internal feature importance measure.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns)
important_features = importances.sort_values(ascending=False)
```

---

### 4. **Permutation Importance (Advanced)**

- Measures drop in model performance when a feature is randomly shuffled.

```python
from sklearn.inspection import permutation_importance
results = permutation_importance(model, X, y, n_repeats=10)
```

---

## 📘 Real-World Example: Titanic

1. **Drop Irrelevant Features**

```python
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
```

2. **Check Feature Importance with Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
```

3. **Apply Lasso for Feature Shrinking**

```python
from sklearn.linear_model import LassoCV
model = LassoCV().fit(X, y)
selected_features = X.columns[model.coef_ != 0]
```

---

## 🧪 Tip: Use Feature Selection with Cross-Validation

Avoid selecting features on the full dataset to prevent data leakage. Use `SelectKBest` or `RFE` inside a pipeline.

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('select', SelectKBest(score_func=chi2, k=5)),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
```

---

## ✅ Outcome of Stage 7:

By the end of this stage:

- You’ve reduced the number of features to the most informative ones
- You avoid overfitting and improve model performance
- You’ve prepared for final model training with optimal inputs

---

Next Step: **Stage 8 - Handling Imbalanced Datasets**

