# 🧪 Stage 8: Sampling, Bootstrapping & Experimental Design

This stage helps you understand how to collect and simulate data properly for statistical testing and model training — a crucial part of robust machine learning pipelines.

---

## 1. Sampling Techniques

### 🎲 Simple Random Sampling

- Every item has equal chance of selection
- Avoids bias, useful for unbiased estimates

📊 *Example*: Pick 100 users randomly from a customer database for a survey

### 📊 Stratified Sampling

- Divide population into strata (e.g., age groups), then sample proportionally
- Ensures key subgroups are represented

📊 *Example*: In a health survey, ensure all age groups have fair representation

### 🔄 Systematic Sampling

- Select every k-th item in the dataset
- Easy to implement, but can introduce patterns

📊 *Example*: Survey every 10th visitor to a website

---

## 2. Bootstrapping (Resampling)

- Technique to **simulate new datasets** by sampling **with replacement**
- Helps estimate **confidence intervals**, **model stability**, and **error bounds**

📊 *Example*: Estimate mean customer purchase by resampling your original data 1,000 times

```python
import numpy as np
boot_means = []
for _ in range(1000):
    sample = np.random.choice(data, size=len(data), replace=True)
    boot_means.append(np.mean(sample))
```

Use `np.percentile(boot_means, [2.5, 97.5])` for 95% confidence interval.

---

## 3. Experimental Design

### 🎯 Control vs Treatment

- Create two groups: control (no change) and treatment (with intervention)
- Compare outcomes to assess causal impact

📊 *Example*: Test whether a new UI increases user engagement

### 🔍 Randomization

- Assign subjects randomly to control/treatment
- Eliminates selection bias

### 📏 Blinding

- Subjects or evaluators don’t know which group is which
- Prevents psychological influence on results

📊 *Example*: A/B test where users don’t know which version they are using

### 🧪 A/B Testing Framework

```python
import scipy.stats as stats

control = np.random.binomial(1, 0.12, 1000)
treatment = np.random.binomial(1, 0.15, 1000)
stat, p = stats.ttest_ind(control, treatment)
print(f"T-statistic = {stat}, p-value = {p}")
```

If p < 0.05, the treatment effect is statistically significant.

---

## ✅ Summary Table

| Concept              | Description                           | Example                              |
| -------------------- | ------------------------------------- | ------------------------------------ |
| Random Sampling      | Equal chance of selection             | Pick random users for survey         |
| Stratified Sampling  | Sample evenly across subgroups        | Fair gender or age representation    |
| Bootstrapping        | Estimate metrics using resampled data | Confidence interval of mean          |
| Control vs Treatment | Isolate effect of change              | New ad design vs old one             |
| Randomization        | Eliminate selection bias              | Assign users to test groups randomly |
| Blinding             | Avoid psychological bias in responses | Testers unaware of version shown     |
| A/B Testing          | Evaluate intervention effectiveness   | Test UI changes or pricing offers    |

---

Next: **Stage 9 – Time Series & Sequential Data** ⏳

