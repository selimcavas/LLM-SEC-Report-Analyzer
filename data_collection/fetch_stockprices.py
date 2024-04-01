import datetime
import sqlite3
import yfinance as yf

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
        if result[0] is not None:
            if datetime.datetime.strptime(result[0], '%Y-%m-%d') == datetime.datetime.now():
                return # If the data is up to date, return
            
            start_date = datetime.datetime.strptime(result[0], '%Y-%m-%d') + datetime.timedelta(days=1)
            
        else:
            start_date = datetime.datetime(2023, 3, 1)  # Default start date

        # Fetch the data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date.strftime('%Y-%m-%d'), end=datetime.datetime.now().strftime('%Y-%m-%d'))

        if not hist.empty:
            # Loop over the historical data and insert each price and date
            for date, row in hist.iterrows():
                price = row['Close']
                date_str = date.strftime('%Y-%m-%d')

                # Insert the data into the database
                print(f'Inserting {ticker} {price} {date_str}')
                c.execute('''
                    INSERT INTO stock_prices (ticker, price, date)
                    VALUES (?, ?, ?)
                ''', (ticker, price, date_str))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()