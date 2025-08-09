from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Importing the data
stocks = pd.read_csv('MicrosoftStock.csv')

# Converting the date column
stocks['date'] = pd.to_datetime(stocks['date'])

# Filtering the data
prediction = stocks.loc[
    (stocks['date'] > datetime(2013,1,1)) & (stocks['date'] < datetime(2018,1,1))
]

# Features and target
features = ['open', 'high', 'low', 'close', 'volume']
target = 'close'

# Separate scalers for features and target
sc_features = StandardScaler()
sc_target = StandardScaler()

scaled_features = sc_features.fit_transform(prediction[features])
scaled_target = sc_target.fit_transform(prediction[[target]])  # double bracket for 2D shape

# Creating sequences
X, y = [], []
time_steps = 60

for i in range(time_steps, len(scaled_features)):
    X.append(scaled_features[i-time_steps:i])  # past 60 days of features
    y.append(scaled_target[i, 0])  # corresponding close price

X, y = np.array(X), np.array(y)

# Train-test split (chronological, no shuffle for time series)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Building the LSTM model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.LSTM(units=64, return_sequences=False))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training
model.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test))

# Predicting
y_pred_scaled = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred = sc_target.inverse_transform(y_pred_scaled)
y_actual = sc_target.inverse_transform(y_test.reshape(-1, 1))

# Plotting
plt.figure(figsize=(10,6))
plt.plot(y_actual, label='Actual Close Price', color='blue')
plt.plot(y_pred, label='Predicted Close Price', color='red')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()
