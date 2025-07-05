# 🧠 Stage 5: Feature Engineering & Selection

In this stage, we shift from analyzing data to actively **transforming** it for machine learning. Great features = great models. We’ll cover how to create meaningful inputs, reduce dimensionality, and select the most impactful variables.

---

## 1. Feature Engineering (Creating Better Inputs)

### 🔄 Transformation Techniques

- **Normalization (Min-Max Scaling)**

  - Scales features to [0, 1]
  - Useful for models like KNN, SVM, neural nets
  - 📊 *Example*: Normalize age (18–65) → [0, 1] for fairness across models

- **Standardization (Z-score Scaling)**

  - Converts features to have mean 0 and std dev 1
  - Useful for PCA, regression
  - 📊 *Example*: Standardize height in cm so it doesn’t dominate smaller features like rating scores

- **Log Transform**

  - Useful when data is skewed (e.g., income, prices)
  - 📊 *Example*: Skewed house prices (like \$50k vs. \$5M) can be made normal using log transform

### 🏷️ Encoding Categorical Variables

- **One-Hot Encoding**

  - 📊 *Example*: Color = [Red, Blue, Green] → Columns: is\_red, is\_blue, is\_green

- **Label Encoding (for tree models)**

  - 📊 *Example*: Encode "Low" = 0, "Medium" = 1, "High" = 2 for ordinal severity levels

### 🧠 Creating Features from Context

- Combine `year` and `month` → `datetime`
- From `purchase_date` → `day_of_week`, `is_weekend`
- From `text` → `word_count`, `keyword_count`

> 🏠 Example: From a real estate dataset:
>
> - Size per room = Total Area / # of rooms
> - Age = Current Year - Built Year

---

## 2. Feature Selection (Choosing the Right Inputs)

### 🎯 Filter Methods

- **Variance Threshold**

  - 📊 *Example*: Drop features like "ZIP Code" if they don’t vary much across records

- **Correlation Filter**

  - 📊 *Example*: Drop one of two highly correlated features like Height vs. Arm Span

### ⚙️ Wrapper Methods

- **Recursive Feature Elimination (RFE)**
  - 📊 *Example*: Run RFE on credit data to retain only most predictive features like income, debt ratio

### 🧪 Embedded Methods

- **Feature Importance from Models**
  - 📊 *Example*: RandomForest ranks features like "number\_of\_logins" higher than "user\_age" for churn prediction

---

## 3. Dimensionality Reduction

### 🧬 PCA (Principal Component Analysis)

- Reduces dimensions by creating new uncorrelated components
- Helps with noise reduction and visualization
- 📊 *Example*: Convert 100 exam questions → 5 principal learning factors

### 📐 t-SNE / UMAP

- Nonlinear methods for visualizing high-dimensional data
- 📊 *Example*: Visualize clusters of handwritten digits (MNIST) in 2D

---

## 4. 🔬 Applied Statistics in Machine Learning

### 📏 Model Assumptions

- **Linearity**
  - 📊 *Example*: Predicting weight from height assumes a linear trend
- **Homoscedasticity**
  - 📊 *Example*: In linear regression, residual errors should have constant variance
- **Normality**
  - 📊 *Example*: For t-tests, assume residuals (errors) follow a normal distribution
- **Independence**
  - 📊 *Example*: Click-through rates of different users should not influence each other

### 📊 Evaluation Metrics (Statistical Perspective)

- **Accuracy**
  - 📊 *Example*: 90 correct predictions out of 100 = 90% accuracy
- **Precision**
  - 📊 *Example*: Spam detection — among emails predicted as spam, how many were actually spam?
- **Recall**
  - 📊 *Example*: Cancer detection — among all actual cancer cases, how many were found?
- **ROC Curve / AUC**
  - 📊 *Example*: Evaluate binary classifier’s performance over all thresholds
- **Log Loss**
  - 📊 *Example*: Punishes high-confidence wrong predictions (e.g., 0.99 probability but was incorrect)

### 🧠 Statistical Significance in ML

- **A/B Testing**
  - 📊 *Example*: Compare click rates between two webpage versions (A vs B)
- **p-values**
  - 📊 *Example*: Determine if a feature like "age" significantly affects income prediction
- **Bootstrap Sampling**
  - 📊 *Example*: Resample dataset many times to estimate model accuracy confidence intervals

### 🧪 Simulated A/B Test Example

```python
import numpy as np
from scipy import stats

a = np.random.binomial(1, 0.6, 100)
b = np.random.binomial(1, 0.55, 100)

stat, p = stats.ttest_ind(a, b)
print(f"T-statistic: {stat}, P-value: {p}")
```

---

## ✅ Summary Table

| Technique               | Purpose                         | Example Use Case           |
| ----------------------- | ------------------------------- | -------------------------- |
| Min-Max Scaling         | Normalize feature ranges        | Image pixel scaling        |
| Z-score Standardization | Normalize for mean & std        | Regression input prep      |
| Log Transform           | Fix skewed distribution         | Price, salary              |
| One-Hot Encoding        | Convert category to numeric     | Color, city                |
| Feature Engineering     | Create more meaningful features | Room size, date parts      |
| Variance/Correlation    | Drop unhelpful features         | Redundant columns          |
| RFE                     | Select top-N predictors         | AutoML, model tuning       |
| PCA                     | Reduce dimensionality           | Visualization, compression |
| A/B Testing             | Test effectiveness              | Compare two ML models      |
| p-values, Bootstrap     | Assess significance, stability  | Feature selection, CI      |

---

Next up: **Stage 6 – Model Evaluation & Validation** 🧪

