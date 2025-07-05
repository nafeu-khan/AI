📊 **Stage 1: Understanding Raw Data (Exploratory Data Analysis - EDA)**

---

### 🎯 Goal:

Before any machine learning can be done, we must understand the structure and quality of the raw data. This stage is all about becoming familiar with the dataset: its shape, feature types, potential targets, and anomalies.

---

## 🧠 Key Concepts:

### 1. **Raw Data**

- Raw data is the unprocessed data collected directly from source systems.
- It may contain missing values, inconsistent types, duplicates, etc.

### 2. **Features and Labels**

- **Features (X):** The columns used to predict an outcome.
- **Label (y):** The target column (what you're trying to predict).

### 3. **EDA (Exploratory Data Analysis)**

- Helps understand data structure, detect anomalies, find patterns.
- Tools: `pandas`, `matplotlib`, `seaborn`, `plotly`

---

## 📘 Real-World Example: Titanic Dataset

- **Goal:** Predict if a passenger survived the Titanic disaster.
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)

### 📁 Sample Columns:

| Column      | Description                       |
| ----------- | --------------------------------- |
| PassengerId | Unique ID                         |
| Pclass      | Ticket class (1 = 1st, etc.)      |
| Name        | Full name                         |
| Sex         | Gender                            |
| Age         | Age in years                      |
| SibSp       | # of siblings / spouses aboard    |
| Parch       | # of parents / children aboard    |
| Fare        | Ticket fare                       |
| Embarked    | Port of Embarkation               |
| Survived    | 0 = No, 1 = Yes (Target variable) |

---

## 🛠️ Step-by-Step: Initial Exploration

### 1. **Load the Data**

```python
import pandas as pd
df = pd.read_csv('titanic.csv')
df.head()
```

### 2. **Dataset Shape and Types**

```python
df.shape          # Rows and columns
df.dtypes         # Data types of each column
df.columns        # List of feature names
```

### 3. **Missing Values**

```python
df.isnull().sum()
```

Identify which features have missing values and how many.

### 4. **Descriptive Stats**

```python
df.describe(include='all')
```

Shows mean, std, min, max, and counts for numerical/categorical data.

### 5. **Check Class Balance (Target)**

```python
df['Survived'].value_counts(normalize=True)
```

Check if the dataset is balanced or skewed.

---

## 📊 Basic Visualization

### 1. **Survival Rate by Sex**

```python
import seaborn as sns
sns.barplot(x='Sex', y='Survived', data=df)
```

### 2. **Age Distribution**

```python
sns.histplot(df['Age'].dropna(), kde=True)
```

### 3. **Correlation Heatmap**

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

---

## ⚠️ Common Issues at This Stage:

- Mixed data types (e.g., numbers stored as strings)
- Categorical features not encoded
- Columns with high missingness
- Duplicates
- Imbalanced target classes

---

## ✅ Outcome of Stage 1:

You should be able to answer:

- What kind of data are you working with?
- What is the target variable?
- What are the potential issues (missing, types, imbalance)?
- What features might need cleaning or transformation?

---

Next Step: **Stage 2 - Handling Missing Data**

