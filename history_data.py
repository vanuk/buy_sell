import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.models import Sequential
import matplotlib.pyplot as plt
import datetime
#from tensorflow.keras.layers import LSTM, Dense, Dropout

def fetch_binance_data(symbol="BTC/USDT", timeframe="5m", limit=500):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

df = fetch_binance_data()
print(df.head())


scaler = MinMaxScaler(feature_range=(0, 1))
df["close_scaled"] = scaler.fit_transform(df["close"].values.reshape(-1,1))

def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 150  # Використовуємо останні 50 точок для прогнозу
X, y = create_sequences(df["close_scaled"].values, seq_length)

split = int(len(X) * 0.8)  # 80% тренувальних, 20% тестових
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Додаємо вивід
print(f"Shape of X_train: {X_train.shape}")  # Очікуваний формат: (samples, seq_length, features)
print(f"Shape of y_train: {y_train.shape}")  # Очікуваний формат: (samples,)
print(f"Shape of X_test: {X_test.shape}")    # Очікуваний формат: (samples, seq_length, features)
print(f"Shape of y_test: {y_test.shape}")    # Очікуваний формат: (samples,)

# Виведемо приклад першої послідовності та очікуваного значення
print("\nExample of first sequence (X_train[0]):")
print(X_train[0].flatten())  # Виводимо в 1D для зручності

print("\nCorresponding target (y_train[0]):")
print(y_train[0])


model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_test, y_test))


predicted = model.predict(X_test)

# Відновлюємо масштаб (inverse transform)
predicted = scaler.inverse_transform(predicted)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Графік реальних та передбачених цін
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label="Actual Price", color='blue')
plt.plot(predicted, label="Predicted Price", color='red')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.show()

future_input = X_test[-1].reshape(1, 150, 1)

# Скільки періодів передбачати
future_steps = 10
future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(future_input)  # Передбачення наступного значення
    future_predictions.append(next_pred[0, 0])  # Додаємо в список
    
    # Оновлюємо вхідні дані: зсуваємо вліво і додаємо нове передбачене значення
    future_input = np.append(future_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

# Відновлюємо масштаб (inverse transform)
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Виводимо передбачені значення
print("Future Predictions:")
for i, pred in enumerate(future_predictions, 1):
    print(f"Step {i}: {pred[0]}")
    
    last_time = datetime.datetime.now()

# Генеруємо часові мітки для майбутніх передбачень з кроком 30 хвилин
future_timestamps = [last_time + datetime.timedelta(minutes=60 * i) for i in range(1, future_steps + 1)]

# Побудова графіка
plt.figure(figsize=(10, 5))
plt.plot(future_timestamps, future_predictions, marker='o', linestyle='-', color='green', label='Future Predictions')

# Налаштування осей
plt.xlabel("Time")
plt.ylabel("Predicted Value")
plt.title("Future Predictions (30 min intervals)")
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Відображення графіка
plt.show()