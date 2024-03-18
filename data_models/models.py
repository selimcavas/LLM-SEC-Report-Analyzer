from pydantic import BaseModel, Field


class TranscriptAnalyzeToolParams(BaseModel):
    ticker: str = Field(
        description="""Given ticker in the prompt taken from user (e.g. AAPL for Apple Inc)""")

    year: str = Field(
        description="""Given year in the prompt taken from user (e.g. 2023)""")

    quarter: str = Field(
        description="""Given quarter in the prompt taken from user in capital letters (e.g Q2) """)


class Text2SQLToolParams(BaseModel):
    text: str = Field(description=""" Prompt from the user that asks for finanical data obtained from the:
                      
            - Yahoo Finance Income Statement Sheet,
            - Yahoo Finance Balance Sheet Sheet,
            - Yahoo Finance Cash Flow Sheet,               
                      
                        """)


class StockPriceVisualizationToolParams(BaseModel):
    prompt: str = Field(
        description="""Prompt from the user that asks for stock price of a company in a given date range.""")
    start_date: str = Field(
        description="""Start date for stock price visualization. In the format YYYY-MM-DD.""")
    end_date: str = Field(
        description="""End date for stock price visualization. In the format YYYY-MM-DD.""")
    ticker: str = Field(
        description="""Ticker for stock price visualization. For example, AAPL for Apple Inc.""")


class CompareStockPriceVisualizationToolParams(BaseModel):
    start_date: str = Field(
        description="""Start date for stock price comparison. In the format YYYY-MM-DD.""")
    end_date: str = Field(
        description="""End date for stock price comparison. In the format YYYY-MM-DD.""")
    ticker1: str = Field(
        description="""First ticker for stock price comparison. For example, AAPL for Apple Inc.""")
    ticker2: str = Field(
        description="""Second ticker for stock price comparison. For example, MSFT for Microsoft Corporation.""")
    prompt: str = Field(
        description="""Prompt from the user that asks for comparison of stock prices and returns of two companies in a given date range.""")
