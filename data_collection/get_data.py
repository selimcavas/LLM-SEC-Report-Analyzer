from traceback import print_tb
from bs4 import BeautifulSoup
from torch import le
import yfinance as yf
import pandas as pd
import requests
import os
import yfinance as yf
import sqlite3
from datetime import datetime, timedelta

# This script gets the financial data from Yahoo Finance and saves it as a CSV file
# It Does this for all the companies in the NASDAQ 100 index
# It takes the last 4 quarters and it gets Income Statement, Balance Sheet and Cash Flow Statement


def get_csvs(ticker):
    # Define the stock symbol of the company you're interested in
    ticker_symbol = ticker  # Replace with the symbol of the company you want to analyze

    # Create a Ticker object
    company = yf.Ticker(ticker_symbol)

    # Get the quarterly financial statements
    quarterly_financials = company.quarterly_financials

    # Filter the income statement for the last 3 quarters
    income_stmt = quarterly_financials

    balance_stmt = company.quarterly_balance_sheet

    cash_stmt = company.quarterly_cashflow

    # Extract the quarter information from the date and use it to create the new column names
    income_stmt.columns = [
        f"{col.year}-Q{col.quarter}" for col in income_stmt.columns]
    balance_stmt.columns = [
        f"{col.year}-Q{col.quarter}" for col in balance_stmt.columns]
    cash_stmt.columns = [
        f"{col.year}-Q{col.quarter}" for col in cash_stmt.columns]

    try:
        financials = pd.concat([income_stmt, balance_stmt, cash_stmt], axis=0)
    except Exception as e:
        return

    # Remove the oldest quarter if there are 5 quarters of data
    if len(financials.columns) >= 5:
        financials = financials.iloc[:, 0:4]

    financials.to_csv(f'data_collection/CSVs/{ticker}.csv')


def get_tickers():
    url = 'https://www.slickcharts.com/nasdaq100'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        soup = BeautifulSoup(res.content, 'html.parser')
        div = soup.find('div', {'class': 'table-responsive'})
        if div is not None:
            table = div.find(
                'table', {'class': 'table table-hover table-borderless table-sm'})
            if table is not None:
                rows = table.tbody.find_all('tr')
                tickers = []
                for row in rows:
                    ticker = row.find_all('td')[2].a.text.strip()
                    tickers.append(ticker)
                return tickers
            else:
                print('Table not found inside div')
                return []
        else:
            print('Div not found on webpage')
            return []
    else:
        print(f'Request failed with status code {res.status_code}')
        print(res.content)
        return []


def write_tickers():
    tickers = get_tickers()
    with open('tickers.txt', 'w') as file:
        for ticker in tickers:
            file.write(ticker + '\n')


def update_stock_prices():
    # Read the tickers from the file
    with open('tickers.txt', 'r') as f:
        tickers = f.read().splitlines()

    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Create the table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker TEXT,
            price REAL,
            volume REAL,
            date TEXT,
            PRIMARY KEY (ticker, date)
        )
    ''')

    # Fetch the stock price data and insert it into the database
    for ticker in tickers:
        # Fetch the latest date for this ticker from the database
        c.execute('''
            SELECT MAX(date) FROM stock_prices WHERE ticker = ?
        ''', (ticker,))
        result = c.fetchone()
        latest_date = result[0] if result else None

        # If there's no latest date, start from 30 days ago
        if latest_date is None:
            start_date = datetime.now() - timedelta(days=2*365)
        else:
            # Start from the day after the latest date
            start_date = datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)

        start_date_str = start_date.strftime('%Y-%m-%d')

        # Fetch the data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date_str)

        if not hist.empty:
            # Loop over the historical data and insert each price and date
            for date, row in hist.iterrows():
                price = row['Close']
                volume = row['Volume']
                date_str = date.strftime('%Y-%m-%d')

                # Insert the data into the database
                c.execute('''
                    INSERT OR IGNORE INTO stock_prices (ticker, price, volume, date)
                    VALUES (?, ?, ?, ?)
                ''', (ticker, price, volume, date_str))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


if __name__ == '__main__':

    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')

    # Get a cursor
    c = conn.cursor()

    # Delete the table
    c.execute('DROP TABLE IF EXISTS stock_prices')


    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    update_stock_prices()