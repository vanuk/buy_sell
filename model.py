import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Завантаження даних (припустимо, ви використовуєте CSV файл)
data = pd.read_csv('BTC-USD.csv')

# Припустимо, що колонка з цінами називається 'Close'
prices = data['close'].values
prices = prices.reshape(-1, 1)  # Перетворюємо вектор в матрицю

# Масштабування даних (нормалізація між 0 і 1)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Створення тренувального набору даних для LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # 60 днів для передбачення наступного
X, y = create_dataset(prices_scaled, time_step)

# Розділяємо дані на тренувальний і тестовий набори
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Перетворюємо X для LSTM (3D формат)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Побудова моделі LSTM
model = keras.Sequential()

# Додавання LSTM шару
model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=50, return_sequences=False))

# Додавання Dense шару для прогнозу
model.add(keras.layers.Dense(units=1))

# Компіляція моделі
model.compile(optimizer='adam', loss='mean_squared_error')

# Навчання моделі
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Прогнозування на тестових даних
y_pred = model.predict(X_test)

# Масштабування результатів назад до оригінального діапазону
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Побудова графіків
plt.plot(y_test, color='blue', label='Real Crypto Prices')
plt.plot(y_pred, color='red', label='Predicted Crypto Prices')
plt.title('Crypto Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
