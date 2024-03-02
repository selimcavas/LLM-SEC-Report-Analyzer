import yfinance as yf
import sqlite3

"""# Connect to the SQLite database
conn = sqlite3.connect('database.db')
c = conn.cursor()

# Delete the table
c.execute('DROP TABLE IF EXISTS stock_prices')

# Commit the changes and close the connection
conn.commit()
conn.close()"""



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
    # Fetch the data
    stock = yf.Ticker(ticker)
    hist =     hist = stock.history(start="2023-03-01", end="2024-03-02")  # Fetch the data for February


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
