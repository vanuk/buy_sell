import ccxt
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.api.models import load_model
import matplotlib.pyplot as plt

def fetch_binance_data(symbol="BTC/USDT", timeframe="1d"):
    exchange = ccxt.binance()
    
    # Отримуємо часові мітки для останніх 2 років
    today = datetime.datetime.now()
    start_date = today - datetime.timedelta(days=730)
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

def create_sequences(data, seq_length=60):  # Використовуємо 60 днів для прогнозу
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(df["close_scaled"].values, seq_length)
if len(X) == 0:
    raise SystemExit("Недостатньо даних для створення навчального набору.")

# Завантажуємо збережену модель
model = load_model('C:/Users/Vanyk/Desktop/crypto_vision/btc_price_model.keras')

# Прогнозуємо на сьогодні
future_input = X[-1].reshape(1, seq_length, 1)
future_steps = 365  # Прогноз на наступні 365 днів
future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(future_input)
    future_predictions.append(next_pred[0, 0])
    future_input = np.append(future_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

# Зворотне масштабування
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Генеруємо часові мітки для прогнозу
last_timestamp = df["timestamp"].iloc[-1]
if last_timestamp.year > datetime.datetime.now().year:
    last_timestamp = datetime.datetime.now()

future_timestamps = [last_timestamp + datetime.timedelta(days=i) for i in range(1, future_steps + 1)]

# Будуємо графік
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["close"], label="Actual Price", color='blue')
plt.plot(future_timestamps, future_predictions, marker='o', linestyle='-', color='red', label='Predicted Price for Next Year')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.title("BTC Price Prediction for Next Year")
plt.xticks(rotation=45)
plt.grid()

# Форматування осей
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
plt.gcf().autofmt_xdate()

plt.show()
