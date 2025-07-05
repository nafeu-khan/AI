# 🤔 Stage 7: Probability & Bayesian Thinking

Probability forms the **mathematical foundation** of how we model uncertainty in machine learning. Bayesian thinking builds on this to reason about model parameters and predictions.

---

## 1. Basics of Probability

- **Experiment**: An action with uncertain outcome (e.g., rolling a die)
- **Sample Space (S)**: All possible outcomes (e.g., {1, 2, 3, 4, 5, 6})
- **Event**: A subset of outcomes (e.g., rolling an even number)
- **Probability**: A number from 0 to 1 that measures likelihood of an event

📊 *Example*: Probability of rolling a 4 = 1/6

---

## 2. Conditional Probability

- Probability of A given B: **P(A | B)**
- Formula: `P(A | B) = P(A ∩ B) / P(B)`

📊 *Example*: Probability of someone having the flu (A) given they have a fever (B)

---

## 3. Bayes’ Theorem

Used to **reverse conditional probability**:

```
P(A | B) = [P(B | A) * P(A)] / P(B)
```

📊 *Example*: Disease diagnosis

- A = has disease
- B = test is positive
- Helps compute true chance of disease given test result

---

## 4. Probability Distributions

### Discrete

- **Bernoulli**: Single yes/no trial (coin flip)
- **Binomial**: # of successes in n trials (e.g., 10 coin flips)

### Continuous

- **Normal Distribution**: Bell-shaped, most data near mean
- **Exponential**: Time between events (e.g., between website clicks)

📊 *Example*:

- Use **binomial** to model success rate of email opens
- Use **normal** to model heights of people

---

## 5. Expectation & Variance

- **Expected Value (Mean)**: Long-run average
- **Variance**: Spread of distribution around mean
- **Standard Deviation**: Square root of variance

📊 *Example*: Expected dice roll = 3.5, Std Dev = \~1.7

---

## 6. Bayesian Inference (Core Idea)

Use data to **update beliefs** about unknowns.

- **Prior**: Belief before data
- **Likelihood**: Data evidence
- **Posterior**: Updated belief

```
Posterior ∝ Prior × Likelihood
```

📊 *Example*: Spam filter

- Prior: 20% emails are spam
- Likelihood: Contains word "free"
- Posterior: Updated chance email is spam

---

## 7. Naive Bayes Classifier (Simple but Powerful)

Assumes feature independence:

```
P(Class | Features) ∝ P(Class) × Π P(Feature_i | Class)
```

📊 *Example*: Text classification

- Predict spam or ham given words in email

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
model.predict(X_test)
```

---

## ✅ Summary Table

| Concept              | Description                     | Example Use Case                  |
| -------------------- | ------------------------------- | --------------------------------- |
| Conditional Prob.    | P(A given B)                    | Fever → Flu likelihood            |
| Bayes Theorem        | Flip conditional probability    | Test accuracy & false positives   |
| Binomial/N.Bernoulli | Model yes/no outcomes           | Email opens / click predictions   |
| Normal Distribution  | Natural continuous data pattern | Heights, IQ, noise                |
| Exponential          | Time between events             | Web click delay, bus arrival time |
| Bayesian Inference   | Update belief using evidence    | Prior → Posterior                 |
| Naive Bayes          | Simple text classification      | Email spam filter                 |

---

Next: **Stage 8 – Sampling, Bootstrapping & Experimental Design** 🧪

