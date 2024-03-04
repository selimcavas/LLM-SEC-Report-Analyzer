from pydantic import BaseModel, Field, FilePath
from typing import Optional



class TranscriptAnalyzeToolParams(BaseModel):
    prompt: str = Field(description="""Prompt from the user that includes keywords such as following:

            - Earning call transcript,
            - Transcript,
                        """)



class Text2SQLToolParams(BaseModel):
    prompt: str = Field(description=""" Prompt from the user that asks for finanical data obtained from the:
                      
            - Yahoo Finance Income Statement Sheet,
            - Yahoo Finance Balance Sheet Sheet,
            - Yahoo Finance Cash Flow Sheet,               
                      
                        """)