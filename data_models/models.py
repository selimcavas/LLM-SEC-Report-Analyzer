from pydantic import BaseModel, Field


class TranscriptAnalyzeToolParams(BaseModel):
    prompt: str = Field(description="""Prompt from the user that must include keywords such as following:

            - Earning call transcript,
            - Transcript,
                        """)


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
