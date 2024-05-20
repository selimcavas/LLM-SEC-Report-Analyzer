import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
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


def get_price_data(ticker, years=2):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=years*365)
    df = pdr.get_data_yahoo(ticker, start, end)
    return df


def getprevious_closest_reports(ticker, date, excel_file="merged_sentiment_scores.xlsx"):
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
                        layers.LSTM(64),
                        # layers.Dropout(0.2),
                        layers.Dense(32, activation='relu', kernel_initializer=initializers.LecunNormal(
                        ), bias_initializer=initializers.LecunNormal()),
                        layers.Dense(1, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01))])
    model.compile(loss='mse',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=200, batch_size=32)
    history = model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=200, batch_size=32)

    # Save the model
    model.save(f'models/lstm_sentiment_filtered/{ticker}.keras')
    return model, history


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
    price_data = get_price_data(ticker, years=5)  # set year here

    # Create a new DataFrame that only contains the 'Close' column
    price_data = price_data[['Close']].copy()

    scaled_data = scaler.fit_transform(price_data)

    price_data = pd.DataFrame(
        scaled_data, columns=['Close'], index=price_data.index)

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

    # Split the data into training, validation, and test sets
    split_index1 = int(len(X) * 0.675)  # 67.5% for training
    split_index2 = int(len(X) * 0.9)  # 22.5% for validation, 10% for testing
    X_train, y_train = X[:split_index1], Y[:split_index1]
    X_test, y_test = X[split_index1:split_index2], Y[split_index1:split_index2]
    X_val, y_val = X[split_index2:], Y[split_index2:]

    # Train the model
    model, history = train_model(X_train, y_train, X_val, y_val, ticker)

    # Plotting the learning curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'],
             label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

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
    new_row.to_csv('filtered_model_evaluations2.csv', mode='a',
                   header=False, index=False)

    # Inverse transform y_test, y_val, and the predictions
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Print out the predicted and actual prices
    for i in range(len(predictions)):
        print(
            f"Predicted price vs actual: {predictions[i][0]}, {y_test[i][0]}")

    # Plot the predicted vs actual prices
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predicted')
    plt.plot(y_test, label='Actual')
    plt.title(f'Predicted vs Actual Prices for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    if not os.path.isfile('filtered_model_evaluations2.csv'):
        eval_data = pd.DataFrame(
            columns=['ticker', 'loss', 'mae', 'mape', 'r2'])
        # Write the DataFrame to an Excel file
        eval_data.to_csv('filtered_model_evaluations2.csv', index=False)

    with open('tickers_test.txt', 'r') as file:
        tickers = file.read().splitlines()

    for ticker in tickers:
        main(ticker)
