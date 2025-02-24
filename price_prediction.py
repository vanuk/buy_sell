import ccxt
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.models import Sequential
import matplotlib.pyplot as plt

def fetch_binance_data(symbol="BTC/USDT", timeframe="15m"):
    exchange = ccxt.binance()
    
    # Отримуємо часові мітки для останніх 30 днів
    today = datetime.datetime.now()
    start_date = today - datetime.timedelta(days=30)
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(today.timestamp() * 1000)
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_timestamp)
        
        if not ohlcv:
            raise ValueError("Binance API не повернув дані!")
        
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        if df.empty:
            raise ValueError("Недостатньо історичних даних!")
        
        return df
    except Exception as e:
        print(f"Помилка отримання даних з Binance: {e}")
        return pd.DataFrame()

df = fetch_binance_data()
if df.empty:
    raise SystemExit("Не вдалося отримати історичні дані.")

# Масштабуємо ціну закриття
scaler = MinMaxScaler(feature_range=(0, 1))
df["close_scaled"] = scaler.fit_transform(df["close"].values.reshape(-1, 1))

def create_sequences(data, seq_length=24):  # Використовуємо 24 години для прогнозу
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 24
X, y = create_sequences(df["close_scaled"].values, seq_length)
if len(X) == 0:
    raise SystemExit("Недостатньо даних для створення навчального набору.")

# Поділ на тренувальні та тестові дані
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Створюємо LSTM-модель
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Тренуємо модель
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test))

# Прогнозуємо на сьогодні
future_input = X_test[-1].reshape(1, seq_length, 1)
future_steps = 24  # Прогноз на наступні 24 години
future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(future_input)
    future_predictions.append(next_pred[0, 0])
    future_input = np.append(future_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

# Зворотне масштабування
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Генеруємо часові мітки для прогнозу
future_timestamps = [df["timestamp"].iloc[-1] + datetime.timedelta(hours=i) for i in range(1, future_steps + 1)]

# Будуємо графік
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["close"], label="Actual Price", color='blue')
plt.plot(future_timestamps, future_predictions, marker='o', linestyle='-', color='red', label='Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.title("BTC Price Prediction for Today")
plt.xticks(rotation=45)
plt.grid()
plt.show()
