# Stage 1: Fundamentals of Statistics for Machine Learning

Welcome to **Stage 1** of your statistics for machine learning journey! This stage focuses on **Descriptive Statistics**, **Types of Data**, and **Data Visualization**. Let’s make it intuitive, visual, and practical!

---

## 🔢 1. Descriptive Statistics

Descriptive statistics help us **summarize and understand** data. Think of it as the first look at your dataset. It answers questions like:

- What does the average data point look like?
- How much do values vary?
- Are there any outliers?

### ✏️ Example Dataset: Daily Sales in a Shop (in USD)

```
[150, 160, 170, 180, 150, 300, 145, 155, 160, 170]
```

### a. **Mean (Average)**

Sum of all values divided by count.

```python
import numpy as np
sales = [150, 160, 170, 180, 150, 300, 145, 155, 160, 170]
mean = np.mean(sales)
print("Mean:", mean)  # Output: 174.0
```

> ✅ Real World: The average daily sales = \$174

### b. **Median (Middle value)**

Value in the middle when sorted. Resistant to outliers.

```python
median = np.median(sales)
print("Median:", median)  # Output: 160.0
```

> 📈 Real World: Half of your days have sales below \$160 and half above.

### c. **Mode (Most Frequent)**

Most occurring value.

```python
from scipy import stats
mode = stats.mode(sales, keepdims=True).mode[0]
print("Mode:", mode)  # Output: 150
```

> 📊 Real World: \$150 was the most common sales value.

### d. **Variance & Standard Deviation**

These measure how spread out the values are from the mean.

- **Variance (σ²)** is the average of the squared differences from the mean.
  - Higher variance means the data is more spread out.
  - Variance is sensitive to extreme values.
- **Standard Deviation (σ)** is the square root of variance.
  - It brings variance back to the original unit, making it easier to interpret.

```python
variance = np.var(sales)
std_dev = np.std(sales)
print("Variance:", variance)     # Output: 1589.0
print("Std Dev:", std_dev)       # Output: 39.86
```

> 📌 **Beginner's Analogy:**
>
> Imagine you and your friends all scored close marks in a test (like 75, 76, 77). The standard deviation is small — everyone performed similarly.
>
> But if scores are 50, 75, 100 — performance is widely spread. Standard deviation is large.

> 🧱 **Real World Applications with Clear Steps:**
>
> **1. Finance (Stock Volatility)**
>
> - **Goal:** Understand how risky a stock is.
> - **How:**
>   1. Get daily stock prices.
>   2. Calculate **daily returns**: (today's price - yesterday's) / yesterday's price.
>   3. Compute standard deviation of these returns.
>   4. Higher std dev = more price fluctuation = higher risk.
> - **Example:** A startup stock might jump wildly → high std dev → risky. A bank stock might stay steady → low std dev → stable.
>
> **2. Manufacturing & Quality Control**
>
> - **Goal:** Maintain consistent product size/quality.
> - **How:**
>   1. Measure product size (e.g., length of screws).
>   2. Calculate mean and standard deviation.
>   3. If std dev is too high, machines may need recalibration.
> - **Example:** Coke bottles must have 500ml ± tolerance. Too much variation = customer complaints.
>
> **3. Education (Exam Analysis)**
>
> - **Goal:** See if students are performing similarly.
> - **How:**
>   1. Collect test scores.
>   2. Calculate std dev per school/class.
>   3. High std dev = mixed performance (some very low, some very high).
> - **Example:** Use std dev to compare performance fairness across different schools.
>
> **4. Machine Learning (Feature Standardization)**
>
> - **Goal:** Prevent large-scaled features from overpowering small ones.
> - **How:**
>   1. Subtract mean and divide by std dev to rescale features.
>   2. Many models like SVM, KNN work better this way.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

> - **Example:** Height in cm (170) vs weight in kg (70) → scaled to same level.
>
> **5. Machine Learning (Model Performance Stability)**
>
> - **Goal:** Evaluate how consistent a model's performance is across different datasets.
> - **How:**
>   1. Train a model multiple times with different train-test splits or cross-validation.
>   2. Record accuracy (or other metric) each time.
>   3. Compute standard deviation of the scores.
>   4. Lower std dev = more stable and reliable model.
>
> ```python
> from sklearn.model_selection import cross_val_score
> from sklearn.ensemble import RandomForestClassifier
>
> model = RandomForestClassifier()
> scores = cross_val_score(model, X, y, cv=5)
> print("Mean Accuracy:", scores.mean())
> print("Stability (Std Dev):", scores.std())
> ```
>
> - **Example:** If one model has std dev = 0.01 and another = 0.15, the first is more stable.

### e. **Range and IQR (Interquartile Range)**

- **Range** = max - min
- **IQR** = Q3 - Q1 (middle 50% of data)

```python
range_ = max(sales) - min(sales)
iqr = np.percentile(sales, 75) - np.percentile(sales, 25)
print("Range:", range_)   # Output: 155
print("IQR:", iqr)        # Output: 20
```

> 🌍 Real World:
>
> - IQR is used in **outlier detection**.
> - In medical research: IQR helps measure the **central concentration** of symptoms or health metrics like blood pressure.

---

## 🌐 2. Types of Data

Understanding **data types** helps you choose the right algorithms and visualizations.

### a. **Qualitative (Categorical)**

- **Nominal**: No order (e.g., gender, color)
- **Ordinal**: Ordered categories (e.g., rating: poor < fair < good)

### b. **Quantitative (Numerical)**

- **Discrete**: Countable (e.g., number of customers)
- **Continuous**: Measurable (e.g., height, weight)

> ✅ Tip: ML models perform best with clean, well-typed numerical features.

---

## 🎭 3. Data Visualization

### a. **Histograms**

Show distribution of continuous data.

```python
import matplotlib.pyplot as plt
plt.hist(sales, bins=5, color='skyblue', edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales ($)')
plt.ylabel('Frequency')
plt.show()
```

> 🔍 Insight: Are your sales clustered around a certain value?

### b. **Boxplots**

Great for spotting outliers.

```python
plt.boxplot(sales)
plt.title('Boxplot of Sales')
plt.show()
```

> 🤔 Notice that 300 might be an outlier?

### c. **Bar Charts** (for categorical)

```python
categories = ['Electronics', 'Clothing', 'Food']
revenues = [1200, 800, 600]
plt.bar(categories, revenues, color='green')
plt.title('Revenue by Category')
plt.ylabel('Revenue ($)')
plt.show()
```

> 🌎 Real World: Which category earns you the most?

---

## ✨ Real-World Example

**Startup User Analysis**: A startup has monthly app usage data:

```python
users = [150, 200, 180, 220, 170, 210, 160, 200, 250, 230]
```

- Mean users: ≈18.
- Are users growing? Plot a line graph.

```python
plt.plot(users, marker='o')
plt.title('Monthly App Users')
plt.xlabel('Month')
plt.ylabel('Users')
plt.grid(True)
plt.show()
```

> 🌟 Conclusion: Use stats to track growth, optimize strategy, and pitch to investors.

---

## 💡 Summary Table

| Metric   | Use Case                       | Formula/Tool      |
| -------- | ------------------------------ | ----------------- |
| Mean     | Average sales                  | `np.mean()`       |
| Median   | Handle outliers                | `np.median()`     |
| Mode     | Most frequent category/value   | `stats.mode()`    |
| Std Dev  | Risk, variation, normalization | `np.std()`        |
| Variance | Data spread/risk analysis      | `np.var()`        |
| IQR      | Range of core data             | `np.percentile()` |
| Boxplot  | Visual outlier check           | `plt.boxplot()`   |

---

## 🚀 What's Next (Stage 2 Preview)

In **Stage 2**, you’ll dive into **probability theory** and **probability distributions** – the backbone of machine learning models like Naive Bayes and Bayesian networks.

---

Ready to continue? Let’s dive into the world of **Probability** next!

