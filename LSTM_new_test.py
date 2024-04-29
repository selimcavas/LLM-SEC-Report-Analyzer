import sqlite3
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import datetime
import datetime as dt
import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def train_and_save_model(ticker):

    if os.path.isfile(f'models/{ticker}.keras'):
        print(f'{ticker} model already exists passing.')
        return

    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Calculate the start date for the last 1 year of data
    # Prepare the SQL query to get the last record date
    sql_query_last_date = '''
        SELECT MAX(date)
        FROM stock_prices
        WHERE ticker = ?
    '''

    # Execute the SQL query
    c.execute(sql_query_last_date, (ticker,))
    end_date = c.fetchall()[0][0]

    print(f"🟢 Last date: {end_date}")

    if end_date == None:
        print(f'{ticker} end date does not exist passing.')
        return

    # Convert the end_date to a datetime object
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    start_date = end_date - relativedelta(years=2)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"🟢 Start date: {start_date_str}, End date: {end_date_str}")

    # Prepare the SQL query
    sql_query = '''
        SELECT date, price, volume
        FROM stock_prices
        WHERE date BETWEEN ? AND ? AND ticker = ?
        ORDER BY date
    '''

    # Execute the SQL query
    c.execute(sql_query, (start_date_str, end_date_str, ticker))
    rows = c.fetchall()

    # Check if any data was fetched
    if not rows:
        return f'No data for {ticker} between {start_date_str} and {end_date_str}'

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(rows, columns=['date', 'price', 'volume'])

    df['price_diff'] = df['price'].diff()

    df.drop(0, inplace=True)
    df.drop('date', axis=1, inplace=True)

    win_length = 5  # window length 5 days
    batch_size = 15  # train
    n_features = 3  # number of features

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    features = scaled_data
    target = scaled_data[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, shuffle=False)

    train_generator = TimeseriesGenerator(
        x_train, y_train, length=win_length, sampling_rate=1, batch_size=batch_size)
    test_generator = TimeseriesGenerator(
        x_test, y_test, length=win_length, sampling_rate=1, batch_size=batch_size)

    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(
        win_length, n_features), return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

    model.fit(train_generator, epochs=30,
              validation_data=test_generator, shuffle=False)

    results = model.evaluate(test_generator)
    print('Loss: ', results[0])
    print('Mean Absolute Error: ', results[1])

    model.save(f'models/{ticker}.keras')


def stock_prices_predictor_tool(days, ticker):
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Calculate the start date for the last 1 year of data
    # Prepare the SQL query to get the last record date
    sql_query_last_date = '''
        SELECT MAX(date)
        FROM stock_prices
        WHERE ticker = ?
    '''

    # Execute the SQL query
    c.execute(sql_query_last_date, (ticker,))
    end_date = c.fetchall()[0][0]

    print(f"🟢 Last date: {end_date}")

    # Convert the end_date to a datetime object
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    start_date = end_date - relativedelta(years=2)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"🟢 Start date: {start_date_str}, End date: {end_date_str}")

    # Prepare the SQL query
    sql_query = '''
        SELECT date, price, volume
        FROM stock_prices
        WHERE date BETWEEN ? AND ? AND ticker = ?
        ORDER BY date
    '''

    # Execute the SQL query
    c.execute(sql_query, (start_date_str, end_date_str, ticker))
    rows = c.fetchall()

    # Check if any data was fetched
    if not rows:
        return f'No data for {ticker} between {start_date_str} and {end_date_str}'

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(rows, columns=['date', 'price', 'volume'])

    df['price_diff'] = df['price'].diff()

    df.drop(0, inplace=True)
    df.drop('date', axis=1, inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    features = scaled_data
    target = scaled_data[:, 0]

    win_length = 5  # window length 5 days
    batch_size = 15  # train
    n_features = 3  # number of features

    pred_list = []
    batch = scaled_data[-win_length:].reshape((1, win_length, n_features))
    print(batch)

    model = load_model(f'models/{ticker}.keras')

    for i in range(days):  # assuming days is the number of days into the future you want to predict
        current_pred = model.predict(batch)[0]
        print(f"🟢 current pred: {current_pred}")
        print(f"🟢 current pred price: {current_pred[0]}")
        # Append only the first element of current_pred
        pred_list.append(current_pred[0])
        current_pred = np.repeat(
            current_pred, n_features).reshape((1, 1, n_features))
        print(f"🟢 batch: {batch[:, 1:, :]}")
        print(f"🟢 current pred reshaped: {current_pred}")
        batch = np.append(batch[:, 1:, :], current_pred, axis=1)

    # Convert pred_list to a numpy array and repeat each prediction 3 times
    pred_array = np.repeat(np.array(pred_list), 3).reshape(-1, 3)
    # Inverse transform the predicted data
    inverse = scaler.inverse_transform(pred_array)
    # Take the first column of inverse as the predictions
    price_preds = inverse[:, 0]
    print(f"🟢 predictions: {price_preds}")


#stock_prices_predictor_tool(3, 'AMZN')
with open('tickers_test.txt', 'r') as f:
    tickers = f.read().splitlines()

for ticker in tickers:
    train_and_save_model(ticker)
