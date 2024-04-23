import sqlite3
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import datetime
import datetime as dt
import pandas as pd
import numpy as np


from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.metrics import RootMeanSquaredError

def stock_prices_predictor_tool(days: str, ticker: str) -> str:
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

    # Calculate the sentiment score
    df['sentiment_score'] = np.sign(df['price_diff']).fillna(0).astype(int)

    print(f"🟢 Fetched data: {df}")

    # Preprocess the data for the LSTM model
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['volume', 'sentiment_score', 'price_diff']])

    n_input = 15
    forward_days = min(int(days), 7)  # User input for number of days, max 7
    n_features = 3  # Adjusted for the new dataframe with added columns


    # Split the data into train and test sets
    train_data, test_data = train_test_split(scaled_data, test_size=0.1, shuffle=False)

    # Create generators for the train and test sets
    train_generator = TimeseriesGenerator(train_data, train_data, length=n_input, batch_size=20)
    test_generator = TimeseriesGenerator(test_data, test_data, length=n_input, batch_size=1)

    # Define the LSTM model
    model = Sequential()

    model.add(LSTM(128, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=False))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss=RootMeanSquaredError())

    model.fit(train_generator, epochs=30, validation_data=test_generator)


    # Use the model to predict future stock prices
    pred_list = []

    current_batch = scaled_data[-n_input:].reshape((1, n_input, n_features))

    print(f"🟢 current_batch: {current_batch}")

    for i in range(forward_days):
        current_pred = model.predict(current_batch)[0]
        print(f"🟢 Current pred: {current_pred}")
        pred_list.append(current_pred)

        current_batch = np.append(current_batch[:, 1:, :], [
                                  [current_pred]], axis=1)

    # Inverse transform the predicted data
    predicted_prices = scaler.inverse_transform(
        np.array(pred_list).reshape(-1, 1))

    # Generate predicted_dates
    last_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    predicted_dates = [(last_date + dt.timedelta(days=i+1)
                        ).strftime('%Y-%m-%d') for i in range(forward_days)]

    # Append predicted prices and dates to rows
    for date, price in zip(predicted_dates, predicted_prices.flatten()):
        rows.append((date, price))

    # Close the connection
    conn.close()

    # Convert the fetched data and predicted data to a list of tuples
    actual_data = list(zip(df['date'].values, df['price'].values))
    predicted_data = list(zip(predicted_dates, predicted_prices.flatten()))

    # Combine the actual and predicted data
    output = actual_data + predicted_data

    # Convert the lists to DataFrames
    actual_data_df = pd.DataFrame(actual_data, columns=['date', 'price'])
    predicted_data_df = pd.DataFrame(predicted_data, columns=['date', 'price'])

    # Calculate the price change
    price_change = (predicted_data_df['price'].values[-1] -
                    actual_data_df['price'].values[-1]) / actual_data_df['price'].values[-1] * 100

    # Get the last actual and predicted dates and prices
    last_actual_date = actual_data_df['date'].values[-1]
    last_predicted_date = predicted_data_df['date'].values[-1]

    last_actual_price = actual_data_df['price'].values[-1]
    last_predicted_price = predicted_data_df['price'].values[-1]

    
    # Convert the output to a DataFrame
    output_df = pd.DataFrame(output, columns=['date', 'prices'])

    # Convert the 'date' column to datetime
    output_df['date'] = pd.to_datetime(output_df['date'])

    # Set the 'date' column as the index
    output_df.set_index('date', inplace=True)

    return output_df