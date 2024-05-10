from edgar import *
from transformers import pipeline, BertTokenizer
import os
import pandas as pd
from sec_parser import parse_10Q_filing

def calculate_sentiment(text):
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    classifier = pipeline('sentiment-analysis', model="yiyanghkust/finbert-tone")

    # Split the text into chunks of 500 tokens or less
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for chunk in chunks:
        result = classifier(chunk)[0]
        sentiment_scores[result['label']] += result['score']

    # Calculate the average sentiment
    avg_sentiment = {k: v / len(chunks) for k, v in sentiment_scores.items()}
    return avg_sentiment

def get_filings_and_sentiment(ticker):
    company = Company(ticker)
    filings = company.get_filings(form="10-Q").latest(2)
    filings_data = []

    for filing in filings:
        filing_text = filing.text()
        filing_text = parse_10Q_filing(filing_text, 4)[0]
        filing_date = filing.filing_date
        sentiment_score = calculate_sentiment(filing_text)
        report_name = f"sec_{ticker}_{filing_date}"
        filings_data.append({
            "ticker": ticker,
            "report_name": report_name,
            "positive_sentiment": sentiment_score['Positive'],
            "negative_sentiment": sentiment_score['Negative'],
            "neutral_sentiment": sentiment_score['Neutral']
        })

    return filings_data

def main():
    with open("tickers_test.txt", "r") as f:
        tickers = f.read().splitlines()
    all_filings_data = []
    for ticker in tickers:
        print(f"Processing {ticker}...")
        filings_data = get_filings_and_sentiment(ticker)
        all_filings_data.extend(filings_data)
    filings_df = pd.DataFrame(all_filings_data)
    filings_df.to_excel("sentiment_scores.xlsx")

if __name__ == "__main__":
    set_identity("Metin metin.arkanoz@ozu.edu.tr")
    main()