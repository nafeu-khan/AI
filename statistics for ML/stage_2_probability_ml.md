# Stage 2: Probability & Distributions for Machine Learning

Welcome to **Stage 2** of your statistics journey! Here we will explore the foundations of **probability**, why it's important, and how **distributions** help machine learning models reason about uncertainty.

---

## 🎲 1. What is Probability?

Probability is a **measure of uncertainty**. It answers the question: *"How likely is something to happen?"*

### ✅ Basic Terms

- **Experiment**: An action with an uncertain outcome (e.g., rolling a die).
- **Sample Space (S)**: All possible outcomes (e.g., {1, 2, 3, 4, 5, 6}).
- **Event**: A subset of outcomes (e.g., rolling an even number).
- **Probability of Event A**:
  ```
  P(A) = (Number of favorable outcomes) / (Total number of outcomes)
  ```

---

## 🧠 2. Real-World Examples

### 🎯 Coin Toss

- S = {Heads, Tails}
- P(Heads) = 0.5

### 🚗 Traffic Prediction

- P(Jam on Monday) = 0.7
- Used in **self-driving cars** to evaluate risk in different routes.

### 📈 Stock Movement

- Predicting rise or fall based on **conditional probability** (Bayes' theorem).

---

## 🔁 3. Types of Probability

### a. **Theoretical Probability**

- Based on reasoning (e.g., dice roll = 1/6)

### b. **Experimental (Empirical) Probability**

- Based on observed data.
- Example:
  ```python
  import numpy as np
  outcomes = np.random.choice(['H', 'T'], size=1000)
  p_heads = np.sum(outcomes == 'H') / 1000
  print("P(Heads):", p_heads)
  ```

### c. **Subjective Probability**

- Personal belief (e.g., "I think there's a 60% chance it will rain today.")

---

## 🔗 4. Conditional Probability & Bayes’ Theorem

> **Conditional Probability**: Probability of A given B happened.

```math
P(A|B) = P(A and B) / P(B)
```

> **Bayes' Theorem**: Used to reverse conditional probabilities.

```math
P(A|B) = [P(B|A) * P(A)] / P(B)
```

### 🔍 Example (Medical Test)

- A disease affects 1% of people.
- Test is 99% accurate.
- What's the probability that a person who tests positive actually has the disease?
- Use Bayes’ theorem to calculate **false positives** and **true positives**.

---

## 📊 5. Probability Distributions

Probability distributions show **how probabilities are spread** over values.

### a. **Discrete Distributions**

- **Bernoulli**: Yes/No, 0/1 events
- **Binomial**: Multiple coin flips (e.g., 5 coin tosses)
- **Poisson**: Count of events in time (e.g., number of calls per hour)

### b. **Continuous Distributions**

#### 🔔 Normal Distribution

- Bell-shaped curve; data is symmetric around the mean.
- **Used when:** Measuring things like height, IQ, test scores.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
plt.plot(x, norm.pdf(x, 0, 1))
plt.title("Normal Distribution (mean=0, std=1)")
plt.grid(True)
plt.show()
```

> 📌 **Real-World Example**:
>
> - Human height: Most people are around the average; few are very tall or short.
> - Sensor noise in electronics follows normal distribution.

---

#### 📏 Uniform Distribution

- All outcomes are equally likely.
- **Used when:** Picking random values within a fixed range.

```python
from scipy.stats import uniform
x = np.linspace(0, 10, 1000)
plt.plot(x, uniform.pdf(x, loc=0, scale=10))
plt.title("Uniform Distribution (0 to 10)")
plt.grid(True)
plt.show()
```

> 📌 **Real-World Example**:
>
> - Random number generator (e.g., pick a number between 1 and 100).
> - Choosing a pixel randomly from a screen.

---

#### ⏳ Exponential Distribution

- Describes time between events in a Poisson process.
- **Used when:** Modeling wait times, e.g., time until next event.

```python
from scipy.stats import expon
x = np.linspace(0, 10, 1000)
plt.plot(x, expon.pdf(x, scale=1))
plt.title("Exponential Distribution")
plt.grid(True)
plt.show()
```

> 📌 **Real-World Example**:
>
> - Time until the next customer arrives in a queue.
> - Time until a radioactive atom decays.

---

## ⚙️ 6. Real-World ML Applications

### 🤖 1. Naive Bayes Classifier

- Based on **Bayes’ Theorem**
- Used in **spam detection**, **sentiment analysis**, **medical diagnosis**

### 🎯 2. Logistic Regression

- Assumes data follows **sigmoid-shaped** probability
- Output = probability of belonging to a class (0 to 1)

### 🎲 3. Data Augmentation & Simulation

- Simulate new samples using probability distributions.

### 💡 4. Anomaly Detection

- Use distributions (e.g., Gaussian) to model normal behavior.
- If a new observation has **very low probability**, it may be an anomaly.

---

## 🧪 7. Practice: Toss Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

n_trials = 1000
results = np.random.choice(['H', 'T'], size=n_trials)
head_ratio = np.sum(results == 'H') / n_trials
print(f"Estimated P(Heads): {head_ratio:.2f}")

plt.bar(['Heads', 'Tails'], [np.sum(results == 'H'), np.sum(results == 'T')])
plt.title('Coin Toss Outcomes')
plt.show()
```

---

## ✅ Summary Table

| Concept              | Use Case                         |
| -------------------- | -------------------------------- |
| Probability          | Forecasting, Risk Modeling       |
| Conditional Prob     | Medical tests, Fraud Detection   |
| Bayes’ Theorem       | Spam filters, Diagnosis tools    |
| Normal Distribution  | Height, IQ, Sensor Noise         |
| Uniform Distribution | Random selections, Simulations   |
| Exponential Dist.    | Waiting times, Customer arrivals |
| Poisson Distribution | Traffic, Web hits, Server loads  |

---

In the next stage, we’ll dive deep into **Inferential Statistics** — learning how to **draw conclusions** about a population using a sample.

Ready to continue to **Stage 3: Inferential Stats**?

