
import datetime
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

import os
import json
from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA

import numpy as np
import pandas as pd
from pinecone import Pinecone
# from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore

# after updates
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain.sql_database import SQLDatabase
from langchain import hub
from langchain.schema.output_parser import StrOutputParser

from sklearn.preprocessing import MinMaxScaler

from langchain_core.prompts import ChatPromptTemplate
# after new scract tool:
import sqlite3
from typing import List

from keras._tf_keras.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import load_model

from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import datetime
import datetime as dt

from sklearn.svm import SVR

from prompts.prompt_templates import transcript_analyze, sql_related, parse_sql, stock_price_chart, cumulative_returns_chart, stock_price_prediction_analysis

# for sentiment analysis and new approachs evaluations
from finbert import calculate_sentiment
from sklearn.metrics import mean_absolute_error

import glob

## importing the util funcs from the utilities.py
from utilities import str_to_datetime, df_to_windowed_df, windowed_df_to_date_X_y, get_price_data, getprevious_closest_reports, df_to_X_y

load_dotenv()
# Bu toollar bir şekilde birden fazla paramater ile çağrılmalı ki böylece args_schema kullanımı anlamlı hale gelsin.

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"


def transcript_analyze_tool(quarter: str, year: str, ticker: str) -> str:
    """
    Used to analyze earning call transcript texts and extract information that could potentially signal risks or growth within a company.

    """

    print('entered transcript tool')
    # Set the environment
    environment = "gcp-starter"

    index_name = "sec-filing-analyzer"

    # Create an instance of the Pinecone class
    pc = Pinecone(api_key=os.environ.get(
        "PINECONE_API_KEY"), environment=environment)

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Use the Pinecone instance to interact with the index
    # docsearch = pc.from_existing_index(index_name, embed)
    text_field = "text"

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name, embed, text_field)

    llm = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY"),
    )

    analyze_prompt = transcript_analyze

    prompt_template = ChatPromptTemplate.from_template(
        analyze_prompt)

    final_response = prompt_template.format_prompt(
        quarter=quarter, year=year, ticker=ticker).to_string()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            # search_type="mmr",
            search_kwargs={"k": 10, 'filter': {'source': f'{ticker.lower()}_{quarter.lower()}_{year}.txt'}}),
    )

    return qa(final_response)


def text2sql_tool(text: str) -> str:
    """
    Used to convert user's prompts to SQL query to obtain financial data.
    """
    chat_model = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    database = SQLDatabase.from_uri(database_uri="sqlite:///database.db")

    template = sql_related

    prompt_template = ChatPromptTemplate.from_template(template)

    response = prompt_template | chat_model | StrOutputParser()

    isRelated = response.invoke({
        "question": text, "table_info": database.get_table_info()
    })

    print("🟢", isRelated)

    if isRelated.strip() == "RELATED":

        few_shot = ""
        with open("prompts/sql_agent_prompts.json", "r") as file:
            few_shot = json.load(file)

        # check if connection is created successfully

        prompt = hub.pull("rlm/text-to-sql")

        # Create chain with LangChain Expression Language
        inputs = {
            "table_info": lambda x: database.get_table_info(),
            "input": lambda x: x["question"],
            "few_shot_examples": lambda x: few_shot,
            "dialect": lambda x: database.dialect,
        }

        sql_response = (
            inputs
            | prompt
            | chat_model.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
        )

        # Call with a given question
        response = sql_response.invoke(
            {"question": str(text)})

        start = response.find('"') + 1

        # Find the end of the SQL query
        end = response.rfind('"')

        # Extract the SQL query
        sql_query = response[start:end]
        print(sql_query)

        # Execute the generated SQL query on the database
        try:
            query_result = database._execute(sql_query)
            parse_template = parse_sql

            parse_temp = ChatPromptTemplate.from_template(parse_template)

            response = parse_temp | chat_model | StrOutputParser()

            query_result = response.invoke(
                {"user_question": text, "query_result": query_result})
        except:
            query_result = "Cannot find the data in the database. Please make sure you provide the correct quarter and year information."

    elif isRelated.strip() == "UNRELATED":
        query_result = """Question is unrelated, please ask something related to the financial data available. Make sure you provide quarter and year information.
        For example:
        
        - Can you list top 5 companies based on the EBITDA data in 2023 q2?
        """
    else:
        query_result = "An error occured in the text2sql tool!"

    return query_result


def stock_prices_visualizer_tool(start: str, end: str, ticker: str):
    '''
    Used to visualize stock prices of a company in a given date range.
    '''
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Prepare the SQL query
    sql_query = '''
        SELECT date, price
        FROM stock_prices
        WHERE date BETWEEN ? AND ? AND ticker = ?
        ORDER BY date
    '''

    # Execute the SQL query
    c.execute(sql_query, (start, end, ticker))
    rows = c.fetchall()

    chart_prompt = stock_price_chart

    prompt_template = ChatPromptTemplate.from_template(chart_prompt)

    chat_model = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    response = prompt_template | chat_model | JsonOutputParser()

    # Close the connection
    conn.close()

    return response.invoke({
        "ticker": ticker, "start": start, "end": end, "rows": rows
    })


def compare_cumulative_returns_tool(start: str, end: str, tickers: List[str]):
    '''
    Used to compare cumulative returns for the stock prices of multiple companies in a given date range. 
    '''
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Prepare the SQL queries
    sql_query = '''
        SELECT date, price
        FROM stock_prices
        WHERE date BETWEEN ? AND ? AND ticker = ?
        ORDER BY date
    '''

    # Execute the SQL query for each ticker and prepare the output
    outputs = []
    for ticker in tickers:
        c.execute(sql_query, (start, end, ticker))
        rows = c.fetchall()

        # Check if any data was fetched
        if not rows:
            return f'No data for {ticker} between {start} and {end}'

        # Prepare the output and calculate cumulative returns
        first_price = rows[0][1]
        output = [
            f'{date}: {((price - first_price) / first_price) * 100 }' for date, price in rows]
        output = "\n".join(output)
        outputs.append(output)

        # Print the outputs
    for i, output in enumerate(outputs):
        print(f"🟢 Output for {tickers[i]}: {output}")

    chart_prompt = cumulative_returns_chart

    prompt_template = ChatPromptTemplate.from_template(chart_prompt)

    chat_model = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    response = prompt_template | chat_model | JsonOutputParser()

    # Close the connection
    conn.close()

    return response.invoke({
        "tickers": tickers, "start": start, "end": end, "output": outputs,
    })


def stock_prices_predictor_tool(ticker):

    price_df = get_price_data(ticker)
    price_data = price_df[['Close']].copy()


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

def svr_prediction(df, forward_days):

    df = df.copy()

    # Preprocess the data for the SVR model
    df['date'] = pd.to_datetime(df['date'])
    original_start_date = df['date'].min()
    df['date'] = (df['date'] - original_start_date).dt.days

    X = df['date'].values.reshape(-1, 1)
    y = df['price'].values

    # Use the MinMaxScaler to scale your data
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))

    # Define the SVR model
    model = SVR(kernel='rbf')

    # Train the model
    model.fit(X, y.ravel())

    # Use the SVR model to predict future stock prices
    future_dates = np.array([(df['date'].max() + i)
                            for i in range(1, forward_days+1)]).reshape(-1, 1)
    predicted_prices_scaled = model.predict(future_dates)
    predicted_prices = scaler.inverse_transform(
        predicted_prices_scaled.reshape(-1, 1))

    # Convert the dates back to datetime format
    future_dates = original_start_date + \
        pd.to_timedelta(future_dates.flatten(), unit='D')

    # Combine the actual and predicted data
    actual_data = df.copy()
    actual_data['date'] = original_start_date + \
        pd.to_timedelta(actual_data['date'], unit='D')
    predicted_data = pd.DataFrame(
        {'date': future_dates, 'price': predicted_prices.flatten()})

    output = pd.concat([actual_data, predicted_data])

    output.set_index('date', inplace=True)
    print(output)

    return predicted_data, output


def get_most_recent_transcript(ticker: str) -> tuple:
    # Define the path to the transcripts folder
    transcripts_folder = os.path.join("data_collection", "transcripts_sample")

    # Create a pattern to match the transcript file name
    pattern = f"{ticker}_*.txt"

    # Get a list of all matching transcript files
    transcript_files = glob.glob(os.path.join(transcripts_folder, pattern))

    # Check if any transcript files were found
    if not transcript_files:
        return f"No transcript found for {ticker}"

    # Extract year and quarter from file names and sort
    sorted_transcript_files = sorted(
        transcript_files,
        key=lambda f: (int(os.path.basename(f).split('_')[2].split('.')[0]), int(
            os.path.basename(f).split('_')[1][1:].replace('q', ''))),
        reverse=True
    )

    # Get the path to the most recent transcript file
    most_recent_transcript = sorted_transcript_files[0]

    # Extract the quarter and year from the file name
    file_name = os.path.basename(most_recent_transcript)
    quarter, year = file_name.split("_")[1].upper(), file_name.split("_")[
        2].split(".")[0]

    # Return the quarter and year separately
    return quarter, year


if __name__ == "__main__":

    stock_prices_predictor_tool("AAPL")