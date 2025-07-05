🔤 **Stage 4: Encoding Categorical Variables**

---

### 🎯 Goal:

Convert categorical data into a numerical format that can be used by machine learning algorithms. Most ML models (except decision trees) require numeric input.

---

## 🧠 Key Concepts:

### 1. **Categorical Features**

These are variables with discrete categories like `Sex`, `Embarked`, `Country`, etc.

---

## 🛠️ Encoding Techniques

### 1. **Label Encoding**

- Assigns an integer to each unique category.
- ⚠️ Risk: Can introduce ordinal relationships where none exist.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])
```

📘 Example:

| Sex    | Encoded |
| ------ | ------- |
| male   | 1       |
| female | 0       |

Use case: Binary categorical variables.

---

### 2. **One-Hot Encoding**

- Creates a binary column for each category.
- Prevents unintended ordinal relationships.

```python
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

📘 Example:

| Embarked | Embarked\_Q | Embarked\_S |
| -------- | ----------- | ----------- |
| C        | 0           | 0           |
| Q        | 1           | 0           |
| S        | 0           | 1           |

Use case: Nominal categorical variables with few unique values.

---

### 3. **Ordinal Encoding**

- Converts categories with natural order into ordered numbers.
- Define the order explicitly.

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['Priority_encoded'] = encoder.fit_transform(df[['Priority']])
```

Use case: Ordered categories like education level, income brackets.

---

### 4. **Frequency / Count Encoding**

- Replaces each category with its count or frequency in the dataset.

```python
freq_map = df['Country'].value_counts().to_dict()
df['Country_freq'] = df['Country'].map(freq_map)
```

Use case: High-cardinality nominal features.

---

### 5. **Target / Mean Encoding**

- Replaces each category with the mean of the target variable for that category.

```python
mean_map = df.groupby('Embarked')['Survived'].mean().to_dict()
df['Embarked_mean'] = df['Embarked'].map(mean_map)
```

⚠️ Use cross-validation to avoid leakage!

---

## 📘 Real-World Example: Titanic Dataset

### Features to Encode:

- `Sex`: Binary → Label Encode
- `Embarked`: Nominal → One-hot Encode
- `Cabin`: High-cardinality → drop or frequency encode

### Practical:

```python
# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

---

## ⚠️ Encoding Tips:

- Always encode after handling missing values.
- Avoid encoding test data with values not seen in training data.
- Use pipelines for consistency.
- For tree-based models, label encoding often suffices.

---

## ✅ Outcome of Stage 4:

By the end of this stage:

- All categorical features are converted to numerical format.
- No leakage or false order introduced.
- Dataset is ready for feature transformation and scaling.

---

Next Step: **Stage 5 - Statistical Thinking for ML**

