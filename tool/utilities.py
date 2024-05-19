import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import os
from tensorflow.keras.models import load_model


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


def get_price_data(ticker, days=90):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
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

def stock_prices_predictor_tool(ticker):

    price_data = get_price_data(ticker)
    price_data = price_data[['Close']].copy()


    # Add the report scores to each row in the price DataFrame
    total_pos = 0
    total_neg = 0
    total_neutral = 0
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
            # Add the sentiment scores to the totals
            total_pos += report_data[j*3]
            total_neg += report_data[j*3 + 1]
            total_neutral += report_data[j*3 + 2]

    # Calculate the average sentiment scores
    avg_pos = total_pos / 4
    avg_neg = total_neg / 4
    avg_neutral = total_neutral / 4

    # Calculate the average sentiment score
    avg_sentiment_score = (avg_pos + avg_neg + avg_neutral) / 3


    X, Y = df_to_X_y(price_data, 10)

    print("🔵", X)
    print("🔴", X.shape)
    
    # Create a StandardScaler instance
    scaler = MinMaxScaler()


    # Fit the scaler to the data and transform the data
    # Reshape X to 2D
    X_2D = X.reshape(-1, X.shape[-1])

    # Scale the data
    X_2D = scaler.fit_transform(X_2D)

    # Reshape X back to 3D
    X = X_2D.reshape(X.shape)

    ## Get the model from models/lstm_sentiment_filtered
    model = load_model(f"models/lstm_sentiment_filtered/{ticker}.keras")
    # Get the predictions
    last_sequence = np.expand_dims(X[-1], axis=0)
    print("🔵", last_sequence)
    predictions = model.predict(last_sequence)
    print("🟢", predictions)
    quit()
    ############################################################################################################

    predicted_dates = [(last_date + dt.timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
    predicted_data = list(zip(predicted_dates, predicted_prices.flatten()))
    
    # Combine the actual and predicted data
    output = price_data + predicted_data

    # Convert the lists to DataFrames
    actual_data_df = pd.DataFrame(price_data, columns=['date', 'price'])
    predicted_data_df = pd.DataFrame(predicted_data, columns=['date', 'price'])

    # Calculate the price change
    price_change = (predicted_data_df['price'].values[-1] -
                    actual_data_df['price'].values[-1]) / actual_data_df['price'].values[-1] * 100

    # Get the last actual and predicted dates and prices
    last_actual_date = actual_data_df['date'].values[-1]
    last_predicted_date = predicted_data_df['date'].values[-1]

    last_actual_price = actual_data_df['price'].values[-1]
    last_predicted_price = predicted_data_df['price'].values[-1]

    template = stock_price_prediction_analysis

    prompt_template = ChatPromptTemplate.from_template(template)

    chat_model = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    trend_change_comment = prompt_template | chat_model | StrOutputParser()

    llm_comment = trend_change_comment.invoke({
        "ticker": ticker,
        "price_change": price_change,
        "last_actual_date": last_actual_date,
        "last_predicted_date": last_predicted_date,
        "last_actual_price": last_actual_price,
        "last_predicted_price": last_predicted_price,
        "sentiment_score": avg_sentiment_score

    }).replace("$", "\$")

    # Convert the output to a DataFrame
    output_df = pd.DataFrame(output, columns=['date', 'prices'])

    # Convert the 'date' column to datetime
    output_df['date'] = pd.to_datetime(output_df['date'])

    # Set the 'date' column as the index
    output_df.set_index('date', inplace=True)

    return output_df, llm_comment

if __name__ == "__main__":
    stock_prices_predictor_tool("AAPL")

