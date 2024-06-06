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
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from torch import mode
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sentiment_supported_LSTM import get_price_data
import os

yf.pdr_override()

scaler = MinMaxScaler()

# Initialize an empty DataFrame
eval_data = pd.DataFrame(
    columns=['ticker', 'loss', 'mae', 'mape', 'r2'])

# Reuse the str_to_datetime, df_to_windowed_df, get_price_data, train_model functions from the previous code


def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n, scaler=scaler):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)

        if len(df_subset) != n+1:
            print(
                f'Warning: Window of size {n} is too large for date {target_date}. Skipping this date.')
        else:
            # values = df_subset.to_numpy()
            values = scaler.transform(df_subset)
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

        next_week = dataframe.loc[target_date:target_date +
                                  datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(
            day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        for j in range(df_subset.shape[1]):
            ret_df[f'Target-{n-i}-{j}'] = X[:, i, j]

    ret_df['Target'] = Y

    return ret_df


def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


def train_model(X_train, y_train, X_val, y_val, ticker):
    model = Sequential([
        layers.Input((10, 1)),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', kernel_initializer=initializers.LecunNormal(
        ), bias_initializer=initializers.LecunNormal()),
        layers.Dense(
            1, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))
    ])
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=200, batch_size=64)
    # history = model.fit(X_train, y_train, validation_data=(
    #     X_val, y_val), epochs=200, batch_size=64)

    # Save the model
    # model.save(f'models/lstm_only/{ticker}.keras')
    return model  # , history


def main(ticker, scaler=scaler):
    global eval_data
    price_data = get_price_data(ticker, years=5)  # set year here

    # Create a new DataFrame that only contains the 'Close' column
    price_data = price_data[['Close']].copy()

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

    print("🔵", X)
    print("🔴", X.shape)
    print("🟢", Y.shape)

    # Split the data into training, validation, and test sets
    split_index1 = int(len(X) * 0.675)  # 67.5% for training
    split_index2 = int(len(X) * 0.9)  # 22.5% for test, 10% for validation
    X_train, y_train = X[:split_index1], Y[:split_index1]
    X_test, y_test = X[split_index1:split_index2], Y[split_index1:split_index2]
    X_val, y_val = X[split_index2:], Y[split_index2:]

    # Reshape y_train, y_val, and y_test to be 2D arrays
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Train the model
    model = train_model(X_train, y_train, X_val, y_val, ticker)

    # Plotting the learning curves
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Losses')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    # plt.plot(history.history['val_mean_absolute_error'],
    #          label='Validation MAE')
    # plt.title('Training and Validation MAE')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

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

    # Plot the predicted vs actual prices
    # plt.figure(figsize=(10, 5))
    # plt.plot(predictions, label='Predicted')
    # plt.plot(y_test, label='Actual')
    # plt.title(f'Predicted vs Actual Prices for {ticker}')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()


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
