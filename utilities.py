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
from prompts.prompt_templates import stock_price_prediction_analysis
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain.schema.output_parser import StrOutputParser

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

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


def get_price_data(ticker, days=30):
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

    last_actual_date = price_data.index[-1]
    last_actual_price = price_data['Close'][-1]

    # last_day_actual_price = price_data

    # Create a StandardScaler instance
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(price_data)

    price_data = pd.DataFrame(
        scaled_data, columns=['Close'], index=price_data.index)

    # Add the report scores to each row in the price DataFrame
    total_pos = 0
    total_neg = 0
    total_neutral = 0
    flag = False
    for i in range(len(price_data)):
        print(
            f"\rProcessing row {i+1}/{len(price_data)} for ticker: {ticker}", end="")

        target_date_in_row = price_data.index[i]
        report_data = getprevious_closest_reports(ticker, target_date_in_row)

        # If there are less than 4 reports, skip this ticker
        if len(report_data) < 12:  # Each report has 3 values (positive, negative, neutral)
            print(f"Skipping {ticker} due to insufficient reports.")
            return
        flag = i == len(price_data) - 1

        for j in range(4):  # There are 4 reports
            price_data.loc[target_date_in_row,
                           f'report_{j}_pos'] = report_data[j*3]
            price_data.loc[target_date_in_row,
                           f'report_{j}_neg'] = report_data[j*3 + 1]
            price_data.loc[target_date_in_row,
                           f'report_{j}_neutral'] = report_data[j*3 + 2]
            if flag:
                # Add the sentiment scores to the totals
                total_pos += report_data[j*3]
                total_neg += report_data[j*3 + 1]
                total_neutral += report_data[j*3 + 2]

    X, Y = df_to_X_y(price_data, 10)

    print("🔵", X)
    print("🔴", X.shape)

    # Get the model from models/lstm_sentiment_filtered
    model = load_model(f"models/lstm_sentiment_filtered/{ticker}.keras")
    # Get the predictions
    last_sequence = np.expand_dims(X[-1], axis=0)
    print("🔵", last_sequence)
    print("🔵🔵", last_sequence[0][0][1])
    predictions = model.predict(last_sequence)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    prediction = predictions[0][0]
    print("🟢", prediction)
    print(last_actual_date)

    # Calculate the price change
    price_change = ((prediction - last_actual_price)/last_actual_price) * 100
    print(price_change)

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
        "last_actual_price": last_actual_price,
        "last_predicted_price": prediction,
        "positive_average_sentiment_score": total_pos / 4,
        "negative_average_sentiment_score": total_neg / 4,
        "neutral_average_sentiment_score": total_neutral / 4

    }).replace("$", "\$")

    print(llm_comment)
    return llm_comment


if __name__ == "__main__":
    stock_prices_predictor_tool("AAPL")
