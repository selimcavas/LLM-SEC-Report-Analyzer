from pydantic import BaseModel, Field, FilePath
from typing import Optional



class TranscriptAnalyzeToolParams(BaseModel):
    prompt: str = Field(description="Prompt from user")



class Text2SQLToolParams(BaseModel):
    text: str = Field(description="Text to be converted to SQL")