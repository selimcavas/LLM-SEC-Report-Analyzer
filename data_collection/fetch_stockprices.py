from datetime import datetime, timedelta
import sqlite3
import yfinance as yf
import streamlit as st


@st.cache_resource
def fetch_and_store_stock_prices():
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
            date TEXT,
            PRIMARY KEY (ticker, date)
        )
    ''')

    # Fetch the stock price data and insert it into the database
    for ticker in tickers:
        # Get the latest date for this ticker
        c.execute('''
            SELECT MAX(date) FROM stock_prices WHERE ticker = ?
        ''', (ticker,))
        result = c.fetchone()

        today = datetime.today().strftime('%Y-%m-%d')

        if result[0] is not None:
            max_date = str(datetime.strptime(result[0], '%Y-%m-%d').date())

            if max_date == today:
                continue  # If the data is up to date, continue

            start_date = datetime.strptime(
                result[0], '%Y-%m-%d').date() + timedelta(days=1)
            # print(f'🟢 start date: {start_date}')

        else:
            print(f'entered else: {ticker}')
            start_date = datetime(2023, 3, 1).date()  # Default start date

        # Fetch the data
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=start_date, end=today)

        # Check if hist is empty
        if not hist.empty:
            # Filter the DataFrame to remove any rows before the start date
            hist = hist[hist.index.date >= start_date]
        else:
            print(f'No data found for {ticker} starting from {start_date}')
            continue

        # print(f'🟣 hist: {hist}')

        if not hist.empty:
            # Loop over the historical data and insert each price and date
            for date, row in hist.iterrows():
                price = row['Close']
                date_str = date.strftime('%Y-%m-%d')

                # Check for duplicates
                c.execute('''
                    SELECT * FROM stock_prices WHERE ticker = ? AND date = ?
                ''', (ticker, date_str))
                result = c.fetchone()

                if result is None:
                    # Insert the data into the database
                    print(f'Inserting {ticker} {price} {date_str}')
                    c.execute('''
                        INSERT INTO stock_prices (ticker, price, date)
                        VALUES (?, ?, ?)
                    ''', (ticker, price, date_str))
        else:
            print(f'Database is up to date for ticker {ticker}!')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
