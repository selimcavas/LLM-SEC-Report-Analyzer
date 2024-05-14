# it will be used to train the LSTM model with only price data.

# the code will be added here later.

import imp
from pyexpat import model
import pandas as pd
import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from torch import mode
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sentiment_supported_LSTM import get_price_data, df_to_windowed_df
import os

yf.pdr_override()

scaler = MinMaxScaler()

# Initialize an empty DataFrame
eval_data = pd.DataFrame(
    columns=['ticker', 'loss', 'mae', 'mape', 'r2'])

# Reuse the str_to_datetime, df_to_windowed_df, get_price_data, train_model functions from the previous code


def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    stock_prices = df_as_np[:, 1:]  # past stock prices

    # Reshape the data to the form (number of samples, timesteps, number of features)
    X_stock_prices = stock_prices.reshape(
        (len(dates), 1, stock_prices.shape[1]))

    X = X_stock_prices

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


def train_model(X_train, y_train, X_val, y_val, ticker):
    model = Sequential([
        layers.Input((1, X_train.shape[2])),  # Changed from 2 to 1
        layers.LSTM(128),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.01),
                  metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    # Save the model
    model.save(f'models/lstm_only/{ticker}.keras')
    return model


def main(ticker, scaler=scaler):
    global eval_data
    price_data = get_price_data(ticker)

    # Fit the scaler on the entire Close price data
    scaler.fit(price_data[['Close']])

    # Convert the price data to a windowed DataFrame
    first_date_str = price_data.index[0].strftime('%Y-%m-%d')
    last_date_str = price_data.index[-1].strftime('%Y-%m-%d')
    windowed_df = df_to_windowed_df(
        price_data, first_date_str, last_date_str, 10, scaler)
    print("✴️", windowed_df.head())
    # Convert the windowed DataFrame to input and target data
    dates, X, Y = windowed_df_to_date_X_y(windowed_df)

    # Split the data into training, validation, and test sets
    split_index1 = int(len(X) * 0.8)
    split_index2 = int(len(X) * 0.9)
    X_train, y_train = X[:split_index1], Y[:split_index1]
    X_val, y_val = X[split_index1:split_index2], Y[split_index1:split_index2]
    X_test, y_test = X[split_index2:], Y[split_index2:]

    # Reshape y_train, y_val, and y_test to be 2D arrays
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Train the model
    model = train_model(X_train, y_train, X_val, y_val, ticker)

    # Get the predictions
    predictions = model.predict(X_test)

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Calculate MAPE and R2 score
    test_mape = mean_absolute_percentage_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)

    # Add the evaluation data to the DataFrame
    new_row = pd.DataFrame({'ticker': [ticker], 'loss': [test_loss], 'mae': [
                           test_mae], 'mape': [test_mape], 'r2': [test_r2]})
    # Append the new row to the CSV file
    new_row.to_csv('only_lstm_model_evaluations.csv', mode='a',
                   header=False, index=False)

    # Inverse transform y_test, y_val, and the predictions
    y_test = scaler.inverse_transform(y_test)
    y_val = scaler.inverse_transform(y_val)
    predictions = scaler.inverse_transform(predictions)

    # Print out the predicted and actual prices
    for i in range(len(predictions)):
        print(
            f"Predicted price vs actual: {predictions[i][0]}, {y_test[i][0]}")


if __name__ == "__main__":

    if not os.path.isfile('only_lstm_model_evaluations.csv'):
        eval_data = pd.DataFrame(
            columns=['ticker', 'loss', 'mae', 'mape', 'r2'])
        # Write the DataFrame to an Excel file
        eval_data.to_csv('only_lstm_model_evaluations.csv', index=False)

    with open('tickers.txt', 'r') as file:
        tickers = file.read().splitlines()

    for ticker in tickers:
        main(ticker)
