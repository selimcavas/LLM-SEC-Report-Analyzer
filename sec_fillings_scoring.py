from edgar import *
from transformers import pipeline, BertTokenizer
import os
import pandas as pd
from sec_parser import parse_10Q_filing

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
classifier = pipeline('sentiment-analysis',
                      model="yiyanghkust/finbert-tone")


def calculate_sentiment(text):
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
    filings = company.get_filings(form="10-Q").latest(20)

    for filing in filings:
        filing_text = filing.text()
        filing_text = parse_10Q_filing(filing_text, 4)[0]
        filing_date = filing.filing_date
        sentiment_score = calculate_sentiment(filing_text)
        report_name = f"sec_{ticker}_{filing_date}"
        filing_data = {
            "ticker": ticker,
            "report_name": report_name,
            "positive_sentiment": sentiment_score['Positive'],
            "negative_sentiment": sentiment_score['Negative'],
            "neutral_sentiment": sentiment_score['Neutral']
        }

        # Check if the Excel file exists
        if not os.path.isfile('sentiment_scores_test.csv'):
            # If not, create a new DataFrame and write it to the Excel file
            df = pd.DataFrame([filing_data])
            df.to_csv('sentiment_scores_test.csv', index=False)
        else:
            # If the Excel file exists, read it into a DataFrame
            df = pd.read_csv('sentiment_scores_test.csv')

            # Check if the report name already exists in the DataFrame
            if report_name in df['report_name'].values:
                # If it does, skip this filing
                continue

            # If the report name does not exist, append the new data to the DataFrame
            df = pd.concat([df, pd.DataFrame([filing_data])],
                           ignore_index=True)
            # Write the DataFrame back to the Excel file
            df.to_csv('sentiment_scores_test.csv', index=False)


def main():

    with open("tickers_test.txt", "r") as f:
        tickers = f.read().splitlines()
    for ticker in tickers:
        print(f"Processing {ticker}...")
        get_filings_and_sentiment(ticker)


if __name__ == "__main__":
    set_identity("Metin metin.arkanoz@ozu.edu.tr")
    main()
