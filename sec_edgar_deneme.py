from transformers import pipeline, BertTokenizer
import os
import glob
import datetime
import pandas as pd

from edgar import *

import re
import unicodedata
from bs4 import BeautifulSoup as bs
import requests
import sys
from sec_parser import parse_10Q_filing

def calculate_sentiment(text):
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    classifier = pipeline('sentiment-analysis', model="yiyanghkust/finbert-tone")

    # Split the text into chunks of 512 tokens or less
    chunks = [text[i:i+510] for i in range(0, len(text), 510)]

    sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for chunk in chunks:
        result = classifier(chunk)[0]
        sentiment_scores[result['label']] += result['score']

    # Calculate the average sentiment
    avg_sentiment = {k: v / len(chunks) for k, v in sentiment_scores.items()}
    return avg_sentiment


def get_filings_and_sentiment(ticker):
    company = Company(ticker)
    filings = company.get_filings(form="10-Q").latest(20)
    filings_texts = []
    sentiment_scores = []
    
    # Create a directory for the filings
    os.makedirs(f'filings/{ticker}', exist_ok=True)
    
    for filing in filings:
        filing_text = filing.text()

        returned_data = parse_10Q_filing(filing_text, 4)[0]
        # print(returned_data)
        # print("-----------------")
        filing_company = filing.company
        filing_date = filing.filing_date
        sentiment_score = calculate_sentiment(filing_text)
        print(f" 🛑 Company: {filing_company}, Date: {filing_date}\nPositive Sentiment Score: {sentiment_score['Positive']}, Negative Sentiment Score: {sentiment_score['Negative']}, Neutral Sentiment Score: {sentiment_score['Neutral']}")

        with open(f"filings/{ticker}/sec_{ticker}_{filing_date.quarter}_{filing_date.year}.txt", "w") as f:
            f.write(filing_text)
    return filings_texts, sentiment_scores

def main():
    with open("tickers_text.txt", "r") as f:
        tickers = f.read().splitlines()
    filings_data = []
    for ticker in tickers:
        filings_texts, sentiment_scores = get_filings_and_sentiment(ticker)
        for i in range(len(filings_texts)):
            filings_data.append({
                "ticker": ticker,
                "filing_text": filings_texts[i],
                "sentiment_score": sentiment_scores[i]
            })
    filings_df = pd.DataFrame(filings_data)
    print(filings_df)

if __name__ == "__main__":
    set_identity("Metin metin.arkanoz@ozu.edu.tr")

    #main()
    
    company = Company("MSFT")
    filings = company.get_filings(form="10-Q").latest(10)

    for filing in filings:
        filing_text = filing.text()

        returned_data = parse_10Q_filing(filing_text, 4)[0]
        # print(returned_data)
        # print("-----------------")
        filing_company = filing.company
        filing_date = filing.filing_date
        sentiment_score = calculate_sentiment(returned_data)
        print(f" 🛑 Company: {filing_company}, Date: {filing_date}\nPositive Sentiment Score: {sentiment_score['Positive']}, Negative Sentiment Score: {sentiment_score['Negative']}, Neutral Sentiment Score: {sentiment_score['Neutral']}")

        print("---------------------------------------------------")
