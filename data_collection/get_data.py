from traceback import print_tb
from bs4 import BeautifulSoup
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
    url = 'https://www.slickcharts.com/nasdaq100'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        soup = BeautifulSoup(res.content, 'html.parser')
        div = soup.find('div', {'class': 'table-responsive'})
        if div is not None:
            table = div.find('table', {'class': 'table table-hover table-borderless table-sm'})
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