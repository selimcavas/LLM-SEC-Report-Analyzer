import numpy as np
import sqlite3
from dateutil.relativedelta import relativedelta
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def stock_prices_predictor_tool(ticker):
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Prepare the SQL query
    sql_query = '''
        SELECT date, price, volume
        FROM (
            SELECT date, price, volume
            FROM stock_prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 30
        )
        ORDER BY date
    '''

    # Execute the SQL query
    c.execute(sql_query, (ticker,))
    rows = c.fetchall()

    # Check if any data was fetched
    if not rows:
        return f'No data for {ticker} in stock prices table.'

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(rows, columns=['date', 'price', 'volume'])

    df['price_diff'] = df['price'].diff()

    df.drop(0, inplace=True)
    df.drop('date', axis=1, inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    win_length = 5  # window length 5 days
    n_features = 3  # number of features

    pred_list = []
    batch = scaled_data[:win_length].reshape((1, win_length, n_features))

    model = load_model(f'models/{ticker}.keras')

    for i in range(len(scaled_data) - win_length):
        current_pred = model.predict(batch)[0]
        pred_list.append(current_pred[0])
        batch = scaled_data[i+1:i+1 +
                            win_length].reshape((1, win_length, n_features))

    # Convert pred_list to a numpy array and repeat each prediction 3 times
    pred_array = np.repeat(np.array(pred_list), 3).reshape(-1, 3)
    # Inverse transform the predicted data
    inverse = scaler.inverse_transform(pred_array)
    # Take the first column of inverse as the predictions
    predicted_prices = inverse[:, 0]

    # Create a DataFrame with actual and predicted prices
    result_df = pd.DataFrame({
        'actual_price': df['price'][win_length:],
        'predicted_price': predicted_prices
    })

    return result_df


# Read the tickers from the file
with open('tickers.txt', 'r') as f:
    tickers = f.read().splitlines()

# Initialize an empty list to store the dataframes
dfs = []

# Loop over the tickers
for ticker in tickers:
    # Call the function for each ticker
    result_df = stock_prices_predictor_tool(ticker)
    # Check if result_df is a DataFrame
    if isinstance(result_df, pd.DataFrame):
        # Add a 'ticker' column to the result_df
        result_df['ticker'] = ticker
        # Append the result_df to the list
        dfs.append(result_df)
    else:
        print(result_df)  # Print the message returned by the function

# Concatenate the dataframes into a single dataframe
combined_df = pd.concat(dfs)

# Reset the index of the combined_df
combined_df.reset_index(drop=True, inplace=True)

# Get the unique tickers
unique_tickers = combined_df['ticker'].unique()

# Initialize an empty DataFrame to store the results
result_df = pd.DataFrame()

# Loop over the unique tickers
for ticker in unique_tickers:
    # Select the rows for the current ticker
    df_ticker = combined_df[combined_df['ticker'] == ticker].copy()

    # Calculate the percentage change between actual and predicted prices
    df_ticker['percentage_change'] = (
        df_ticker['predicted_price'] - df_ticker['actual_price']) / df_ticker['actual_price']

    # Scale the percentage changes by a factor of 10
    df_ticker['percentage_change'] *= 10

    # Scale the percentage changes to the range [-1, 1]
    df_ticker['sentiment_score'] = df_ticker['percentage_change'].apply(
        lambda x: max(min(x, 1), -1))

    # Drop the 'percentage_change' column as it's no longer needed
    df_ticker.drop('percentage_change', axis=1, inplace=True)

    # Append the df_ticker to the result_df
    result_df = pd.concat([result_df, df_ticker])

# Reset the index of the result_df
result_df.reset_index(drop=True, inplace=True)

print(result_df)

# Prepare your data
X = result_df[['actual_price', 'sentiment_score']]
y = result_df['predicted_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_absolute_error',
              optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# Save the model
model.save('models/perceptron.keras')
