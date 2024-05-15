import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os

yf.pdr_override()

scaler = MinMaxScaler()

# Initialize an empty DataFrame
eval_data = pd.DataFrame(
    columns=['ticker', 'loss', 'mae', 'mape', 'r2'])


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

    stock_prices = df_as_np[:, 1:11]  # 10 past stock prices
    sentiment_scores = df_as_np[:, 11:-1]  # 12 sentiment scores

    # Reshape the data to the form (number of samples, timesteps, number of features)
    X_stock_prices = stock_prices.reshape(
        (len(dates), 1, stock_prices.shape[1]))
    X_sentiment_scores = sentiment_scores.reshape(
        (len(dates), 1, sentiment_scores.shape[1]))

    # Concatenate the stock prices and sentiment scores along the time step axis
    X = np.concatenate((X_stock_prices, X_sentiment_scores), axis=2)

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


def get_price_data(ticker, years=2):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=years*365)
    df = pdr.get_data_yahoo(ticker, start, end)
    return df


def getprevious_closest_reports(ticker, date, excel_file="filtered_sentiment_scores.xlsx"):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file)

    # Filter out rows where the ticker does not match the input ticker
    df = df[df['ticker'] == ticker]

    # Extract the dates from the report names
    df['date'] = pd.to_datetime(df['report_name'].str.split('_').str[-1])

    # Filter out reports that are after the input date
    df = df[df['date'] < pd.to_datetime(date)]

    # Calculate the absolute difference between the input date and the dates in the DataFrame
    df['date_diff'] = (pd.to_datetime(date) - df['date']).abs()

    # Sort the DataFrame by the date difference and get the 4 closest reports
    closest_reports = df.sort_values('date_diff').head(4)

    # Return the relevant columns as a list of dictionaries
    return closest_reports[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']].values.flatten().tolist()


def train_model(X_train, y_train, X_val, y_val, ticker):
    model = Sequential([layers.Input((X_train.shape[1], X_train.shape[2])),
                        layers.LSTM(128),
                        layers.Dense(64, activation='relu'),
                        layers.Dense(1)])
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.01),
                  metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    # Save the model
    model.save(f'models/lstm_sentiment_filtered/{ticker}.keras')
    return model


def df_to_X_y(df, window_size=10):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][0]
        y.append(label)
    return np.array(X), np.array(y)


def main(ticker, scaler=scaler):
    global eval_data
    price_data = get_price_data(ticker)

    # Create a new DataFrame that only contains the 'Close' column
    price_data = price_data[['Close']].copy()

    # Add the report scores to each row in the price DataFrame
    for i in range(len(price_data)):
        print(
            f"\rProcessing row {i+1}/{len(price_data)} for ticker: {ticker}", end="")

        target_date_in_row = price_data.index[i]
        report_data = getprevious_closest_reports(ticker, target_date_in_row)

        # If there are less than 4 reports, skip this ticker
        if len(report_data) < 12:  # Each report has 3 values (positive, negative, neutral)
            print(f"Skipping {ticker} due to insufficient reports.")
            return

        for j in range(4):  # There are 4 reports
            price_data.loc[target_date_in_row,
                           f'report_{j}_pos'] = report_data[j*3]
            price_data.loc[target_date_in_row,
                           f'report_{j}_neg'] = report_data[j*3 + 1]
            price_data.loc[target_date_in_row,
                           f'report_{j}_neutral'] = report_data[j*3 + 2]

    X, Y = df_to_X_y(
        price_data, 10)

    print("🔵", X)
    print("🔴", X.shape)
    print("🟢", Y)
    print("🟢", Y.shape)

    # Create a StandardScaler instance
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    # Fit the scaler to the data and transform the data
    # Reshape X to 2D
    X_2D = X.reshape(-1, X.shape[-1])

    # Scale the data
    X_2D = scaler.fit_transform(X_2D)

    # Reshape X back to 3D
    X = X_2D.reshape(X.shape)
    Y = scaler_Y.fit_transform(Y.reshape(-1, 1))

    # Split the data into training, validation, and test sets
    split_index1 = int(len(X) * 0.8)
    split_index2 = int(len(X) * 0.9)
    X_train, y_train = X[:split_index1], Y[:split_index1]
    X_val, y_val = X[split_index1:split_index2], Y[split_index1:split_index2]
    X_test, y_test = X[split_index2:], Y[split_index2:]

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
    new_row.to_csv('filtered_model_evaluations.csv', mode='a',
                   header=False, index=False)

    # Inverse transform y_test, y_val, and the predictions
    y_test = scaler_Y.inverse_transform(y_test)
    y_val = scaler_Y.inverse_transform(y_val)
    predictions = scaler_Y.inverse_transform(predictions)

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

    if not os.path.isfile('filtered_model_evaluations.csv'):
        eval_data = pd.DataFrame(
            columns=['ticker', 'loss', 'mae', 'mape', 'r2'])
        # Write the DataFrame to an Excel file
        eval_data.to_csv('filtered_model_evaluations.csv', index=False)

    with open('tickers.txt', 'r') as file:
        tickers = file.read().splitlines()

    for ticker in tickers:
        main(ticker)
