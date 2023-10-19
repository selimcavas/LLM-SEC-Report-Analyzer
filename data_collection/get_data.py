import yfinance as yf

# Define the stock symbol of the company you're interested in
ticker_symbol = 'AAPL'  # Replace with the symbol of the company you want to analyze

# Create a Ticker object
company = yf.Ticker(ticker_symbol)

# Get the earnings data
print(company.income_stmt)
