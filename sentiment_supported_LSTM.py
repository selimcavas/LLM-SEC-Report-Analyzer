import pandas as pd
import datetime
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
yf.pdr_override()

scaler = MinMaxScaler()


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
            # values = df_subset['Close'].to_numpy()
            values = scaler.transform(df_subset[['Close']])
            x, y = values[:-1], values[-1][0]

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
        ret_df[f'Target-{n-i}'] = X[:, i]

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


def getprevious_closest_reports(ticker, date, excel_file="sentiment_scores.xlsx"):
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


def train_model(X_train, y_train, X_val, y_val):
    model = Sequential([layers.Input((2, X_train.shape[2])),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.01),
                  metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    return model


def main(ticker, scaler=scaler):
    price_data = get_price_data(ticker)

    # Fit the scaler on the entire Close price data
    scaler.fit(price_data[['Close']])

    # Convert the price data to a windowed DataFrame
    first_date_str = price_data.index[0].strftime('%Y-%m-%d')
    last_date_str = price_data.index[-1].strftime('%Y-%m-%d')
    windowed_df = df_to_windowed_df(
        price_data, first_date_str, last_date_str, 10)
    print("✴️", windowed_df.head())
    # Add the report scores to each row in the windowed DataFrame
    for i in range(len(windowed_df)):
        target_date_in_row = windowed_df.iloc[i, 0]
        report_data = getprevious_closest_reports(ticker, target_date_in_row)

        for j in range(4):  # There are 4 reports
            windowed_df.loc[i, f'report_{j}_pos'] = report_data[j*3]
            windowed_df.loc[i, f'report_{j}_neg'] = report_data[j*3 + 1]
            windowed_df.loc[i, f'report_{j}_neutral'] = report_data[j*3 + 2]

    # Rearrange the columns
    cols = windowed_df.columns.tolist()
    # Move 'Target' column to the end
    cols = cols[:11] + cols[12:] + [cols[11]]
    windowed_df = windowed_df[cols]

    print("🛑", windowed_df.head())

    # Convert the windowed DataFrame to input and target data
    dates, X, Y = windowed_df_to_date_X_y(windowed_df)

    # Split the data into training, validation, and test sets
    split_index1 = int(len(X) * 0.8)
    split_index2 = int(len(X) * 0.9)
    X_train, y_train = X[:split_index1], Y[:split_index1]
    X_val, y_val = X[split_index1:split_index2], Y[split_index1:split_index2]
    X_test, y_test = X[split_index2:], Y[split_index2:]
    # X_train = X_train.reshape(X_train.shape[0], 2, -1)
    # X_val = X_val.reshape(X_val.shape[0], 2, -1)
    # X_test = X_test.reshape(X_test.shape[0], 2, -1)

    # Reshape y_train, y_val, and y_test to be 2D arrays
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Fit the scaler on y_train and transform y_train, y_val, and y_test
    # scaler.fit(y_train)
    # y_train = scaler.transform(y_train)
    # y_val = scaler.transform(y_val)
    # y_test = scaler.transform(y_test)

    print("🔵", X_train)
    print("🔴", X_train.shape)
    print("🟢", y_train.shape)
    # Train the model
    model = train_model(X_train, y_train, X_val, y_val)

    # Get the predictions
    predictions = model.predict(X_test)

    # Evaluate the model on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Inverse transform y_test, y_val, and the predictions
    y_test = scaler.inverse_transform(y_test)
    y_val = scaler.inverse_transform(y_val)
    predictions = scaler.inverse_transform(predictions)

    # Print out the predicted and actual prices
    for i in range(len(predictions)):
        print(
            f"Predicted price vs actual: {predictions[i][0]}, {y_test[i][0]}")


if __name__ == "__main__":
    main('AAPL')
