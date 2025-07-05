🧹 **Stage 3: Cleaning & Fixing Data**

---

### 🎯 Goal:

To ensure the dataset is logically consistent, clean, and free of anomalies that may hinder model performance. This includes fixing incorrect types, duplicates, outliers, and inconsistent values.

---

## 🧠 Key Concepts:

### 1. **Data Type Conversion**

- Convert columns to appropriate data types to save memory and ensure correct operations.

```python
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df['Sex'] = df['Sex'].astype('category')
df['Survived'] = df['Survived'].astype(int)
```

### 2. **Removing Duplicates**

- Duplicate records can bias the model.

```python
df.drop_duplicates(inplace=True)
```

### 3. **Fixing Typos & Inconsistent Categories**

- Strings with inconsistent spelling or casing need standardization.

```python
df['Sex'] = df['Sex'].str.strip().str.lower().replace({'femail': 'female'})
```

### 4. **Handling Out-of-Range or Illogical Values**

- For example, negative ages or fares.

```python
df = df[df['Age'] >= 0]  # Age should not be negative
```

### 5. **Cap/Floor Outliers (Winsorization)**

- Treat extreme values based on domain or statistical boundaries (IQR method).

```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]
```

---

## 📘 Real-World Example: Titanic Dataset

### Cleaning Steps:

1. **Remove Irrelevant Features**:

```python
df.drop(columns=['Ticket', 'Name'], inplace=True)  # These don't help model prediction
```

2. **Standardize Categorical Columns**:

```python
df['Embarked'] = df['Embarked'].str.upper().fillna('S')
df['Sex'] = df['Sex'].str.lower()
```

3. **Fix Age Outliers**:

```python
# Drop ages above 80 (unusual in Titanic context)
df = df[df['Age'] <= 80]
```

4. **Trim Whitespaces in Strings**:

```python
str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
```

5. **Check Duplicates & Drop**:

```python
df.drop_duplicates(inplace=True)
```

---

## 🛠️ Additional Cleaning Tips:

- Use `.value_counts()` to spot inconsistent categorical labels.
- Visualize distributions (`histplot`, `boxplot`) to find odd patterns.
- For temporal data, ensure datetime formats are correctly parsed.

```python
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
```

---

## ⚠️ Common Data Quality Issues:

| Issue       | Example              | Fix                      |
| ----------- | -------------------- | ------------------------ |
| Wrong type  | 'Age' as object      | Use `pd.to_numeric`      |
| Typos       | 'femail' vs 'female' | `.replace()` or `.map()` |
| Whitespaces | ' male ' vs 'male'   | `.str.strip()`           |
| Outliers    | Fare = 512           | Use IQR method           |
| Duplicates  | Repeated rows        | `.drop_duplicates()`     |

---

## ✅ Outcome of Stage 3:

By the end of this stage, your dataset should:

- Have correct data types
- Be free from duplicates
- Contain logically valid values
- Be ready for encoding and feature transformation

---

Next Step: **Stage 4 - Encoding Categorical Variables**

