# ⏳ Stage 9: Time Series & Sequential Data

Time series data appears when observations are **ordered in time** — like stock prices, sensor readings, or weather. Machine learning on such data requires techniques that respect temporal structure.

---

## 1. Characteristics of Time Series

- **Temporal Order Matters**: The order of data points is crucial because past values influence future ones. Breaking this order (like shuffling) would destroy the meaning.

- **Trend**: A long-term increase or decrease in the data. It reflects gradual changes over time, such as rising temperatures due to climate change.

- **Seasonality**: Patterns that repeat at regular intervals — like daily, weekly, monthly, or yearly. For example, ice cream sales peaking in summer.

- **Stationarity**: A stationary time series has constant mean, variance, and autocorrelation over time. Many time series methods assume stationarity for accurate forecasting.

📊 *Example*: Electricity usage rises during evenings (daily seasonality), and increases steadily over years (trend). Before modeling, we often remove trend and seasonality to make the data stationary.

---

## 2. Time Series Preprocessing

### 🔄 Lag Features

- Create new features using previous time steps. This allows models to learn from history.

📊 *Example*: To predict today’s temperature, use temperatures from yesterday and the day before as features.

```python
df['lag_1'] = df['temp'].shift(1)  # yesterday's temp
df['lag_2'] = df['temp'].shift(2)  # two days ago
```

### 📉 Differencing

- Subtract previous values to remove trend and make data stationary.

📊 *Example*: If sales today are 110 and yesterday 100, the differenced value is 10. This helps reduce non-stationary effects.

```python
df['diff'] = df['value'] - df['value'].shift(1)
```

### 🧼 Rolling Statistics

- Apply moving average or standard deviation over a sliding window to smooth the data or create new features.

📊 *Example*: Use a 3-day moving average to see smoothed temperature trends.

```python
df['rolling_mean'] = df['value'].rolling(window=3).mean()
```

---

## 3. Time-Based Train/Test Split

- Randomly shuffling time series data breaks the time dependency. Instead, split the data **chronologically**.

📊 *Example*: For a stock prediction model:

- Train: 2010–2018
- Test: 2019–2020

```python
train = df[df['year'] <= 2018]
test = df[df['year'] > 2018]
```

---

## 4. Forecasting Methods

### 🔮 ARIMA (AutoRegressive Integrated Moving Average)

- Combines:
  - **AR**: Regression on previous values
  - **I**: Differencing to remove trend
  - **MA**: Regression on past forecast errors

📊 *Example*: Forecasting sales where each month's sales depends on the last 3 months and some noise.

### 🧠 Prophet (by Facebook)

- Automatically models:
  - Trend (linear or logistic)
  - Seasonality (daily, weekly, yearly)
  - Holidays/events (customizable)
- Works well with missing data and outliers.

📊 *Example*: Predict website traffic with spikes during holidays and weekends.

```python
from prophet import Prophet
model = Prophet()
model.fit(df)
forecast = model.predict(future)
```

### 🔁 LSTM (Long Short-Term Memory)

- A type of **recurrent neural network (RNN)** that learns long-term dependencies in sequential data.
- Good for non-linear, noisy time series.

📊 *Example*: Predict future stock prices using deep learning that captures complex patterns over time.

---

## 5. Time Series Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average of absolute differences between predicted and actual values. Easy to interpret.
- **RMSE (Root Mean Squared Error)**: Like MAE, but penalizes large errors more.
- **MAPE (Mean Absolute Percentage Error)**: Shows errors as a percentage — useful for business users.
- **Cross-validation**: Instead of random folds, use **time-based folds**:
  - **Expanding window**: Start small and expand training data each time.
  - **Walk-forward**: Slide training and test windows forward.

📊 *Example*: Evaluate how well sales forecast model generalizes month after month.

---

## ✅ Summary Table

| Concept       | Description                      | Example Use Case               |
| ------------- | -------------------------------- | ------------------------------ |
| Lag Features  | Use past values as inputs        | Predict tomorrow’s weather     |
| Differencing  | Remove trends                    | Stationarize GDP series        |
| Rolling Stats | Smooth noisy data                | Moving average of heart rate   |
| Time Split    | Chronological data partition     | Train: 2010–2019, Test: 2020   |
| ARIMA         | Classic statistical forecasting  | Monthly product sales forecast |
| Prophet       | Robust to trend + seasonality    | Web traffic with holidays      |
| LSTM          | Deep learning for sequences      | Stock price prediction         |
| MAE/RMSE/MAPE | Evaluate time series predictions | Forecasting energy usage       |

---

Next: **Stage 10 – Statistical Thinking in Real ML Projects** 🎯

