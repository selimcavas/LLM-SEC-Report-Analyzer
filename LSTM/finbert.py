
from transformers import pipeline, BertTokenizer
import os
import glob
import datetime


def calculate_sentiment(text):
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    classifier = pipeline('sentiment-analysis', model="ProsusAI/finbert")

    # Split the text into chunks of 512 tokens or less
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]

    sentiment_scores = []
    for chunk in chunks:
        result = classifier(chunk)[0]
        sentiment_score = result['score'] if result['label'] == 'positive' else -result['score']
        sentiment_scores.append(sentiment_score)

    # Calculate the average sentiment
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    return avg_sentiment


if __name__ == "__main__":

    report_text = """For the second quarter of 2022, Apple Inc. (AAPL) reported a revenue of $97.27 billion, representing a 9% increase compared to Q2 2021. The net income stood at $25.0 billion, up by 6% YoY. The company's cash flow from operating activities was $28.06 billion, a 10% increase compared to Q2 2021.
    Detailed Analysis of Financial Performance

        iPhone Segment: The iPhone segment continued to generate the majority of Apple's revenue, with Q2 2022 revenue at $50.57 billion, a 5% YoY increase.
        Services Segment: The Services segment saw significant growth, with revenue of $19.82 billion, a 17% YoY increase.
        Mac Segment: The Mac segment reported revenue of $10.44 billion, a 14% YoY increase.
        iPad Segment: The iPad segment saw a decline in revenue, with Q2 2022 revenue at $7.65 billion, a 2% YoY decrease.
        Wearables, Home, and Accessories Segment: This segment reported revenue of $8.8 billion, up by 12% YoY.

    Key Points from Earnings Call

        Supply Chain Constraints:
            Importance: Supply chain constraints have been a challenge for Apple, affecting its ability to meet consumer demand.
            Excerpt: "We've been able to navigate through the supply constraints with minimal impact to our lineup." - Luca Maestri, CFO.
            Relevant Metric: No significant impact on revenue reported.

        Growth in Services Segment:
            Importance: The Services segment is a key area of growth for Apple, contributing to its recurring revenue stream.
            Excerpt: "Revenue from our Services category reached an all-time high of $19.8 billion, up 17% over last year." - Tim Cook, CEO.
            Relevant Metric: Services revenue growth of 17% YoY.

        Growth in Greater China:
            Importance: Greater China is a critical market for Apple, accounting for a significant portion of its revenue.
            Excerpt: "Greater China revenue was $18.3 billion, up 29% over last year." - Tim Cook, CEO.
            Relevant Metric: 29% YoY growth in Greater China revenue.

    Future Outlook

    Apple's focus on services, new product categories, and expansion in emerging markets like India and Africa position it for continued growth. However, ongoing supply chain constraints and potential economic downturns present risks to its financial performance.
    Conclusion

    Based on the Q2 2022 financial performance and statements made during the earnings call, Apple is on a growth trajectory. The company's diverse product portfolio, focus on services, and strategic expansion into new markets outweigh the risks associated with supply chain constraints and potential economic downturns."
    """
    sentiment_score = calculate_sentiment(report_text)

    print(f"Sentiment Score: {sentiment_score}")
