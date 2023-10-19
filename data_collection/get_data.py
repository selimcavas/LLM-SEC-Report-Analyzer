from traceback import print_tb
from torch import le
import yfinance as yf
import pandas as pd
import requests
import os

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


    financials = pd.concat([income_stmt, balance_stmt, cash_stmt], axis=0)

    financials.to_csv(f'data_collection/CSVs/{ticker}.csv')


def get_tickers():
    headers={"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"}
    res=requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",headers=headers)
    main_data=res.json()['data']['data']['rows']
    tickers = []

    for i in range(len(main_data)):
        tickers.append(main_data[i]['symbol'])

    # print(len(tickers))
    # # sort the tickers alphabetically

    # tickers.sort()
    # # Write the tickers to a text file
    # try:
    #     with open('tickers.txt', 'w') as f:
    #         for ticker in tickers:
    #             f.write(ticker + '\n')
    #     print(f'Successfully wrote {len(tickers)} tickers to tickers.txt')
    # except Exception as e:
    #     print(f'Error writing tickers to file: {e}')
    
    return tickers

if __name__ == '__main__':
    tickers = get_tickers()
    for ticker in tickers:
        try:
            get_csvs(ticker)
        except Exception as e:
            print(e)
            print_tb(e.__traceback__)
            print(ticker)
            continue