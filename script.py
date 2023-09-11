# Optimize this Python script:

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Define constants
SYMBOL = 'AAPL'
START_DATE = '2010-01-01'
END_DATE = '2021-12-31'
TRAIN_RATIO = 0.7
EPOCHS = 50
BATCH_SIZE = 1


def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date).fillna(0)
    return data


def preprocess_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values)
    train_size = int(len(data_scaled) * TRAIN_RATIO)
    train_data = data_scaled[:train_size + 1]
    test_data = data_scaled[train_size:]
    return train_data, test_data, scaler


def create_features(data):
    return np.concatenate((data[:-1], data[1:]), axis=1)


def train_model(X_train, y_train, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch_size, verbose=2)
    return model


def evaluate_model(model, data, scaler):
    predictions = scaler.inverse_transform(model.predict(data))
    rmse = np.sqrt(np.mean(np.square(predictions - data[:, 0])))
    return rmse


def predict_symbols(predictions, data):
    symbols = np.where(predictions > data[:, 0], 'Buy', 'Sell')
    return symbols


def plot_symbols(predicted_symbols, actual_symbols):
    plt.plot(predicted_symbols, label='Predicted Symbols')
    plt.plot(actual_symbols, label='Actual Symbols')
    plt.xlabel('Time')
    plt.ylabel('Symbols')
    plt.title('Predicted vs Actual Symbols')
    plt.legend()
    plt.show()


def main():
    # Fetch historical stock data
    data = fetch_stock_data(SYMBOL, START_DATE, END_DATE)

    # Data preprocessing
    train_data, test_data, scaler = preprocess_data(data)

    # Feature engineering
    X_train = create_features(train_data)
    X_test = create_features(test_data)
    y_train = train_data[1:, 0]
    y_test = test_data[1:, 0]

    # Build a prediction model using deep learning
    model = train_model(X_train, y_train, EPOCHS, BATCH_SIZE)

    # Model evaluation
    train_rmse = evaluate_model(model, X_train, scaler)
    test_rmse = evaluate_model(model, X_test, scaler)

    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Symbol prediction
    predicted_symbols = predict_symbols(model.predict(X_test), test_data)
    actual_symbols = predict_symbols(y_test, test_data)

    # Plotting the predicted and actual symbols
    plot_symbols(predicted_symbols, actual_symbols)


if __name__ == "__main__":
    main()
