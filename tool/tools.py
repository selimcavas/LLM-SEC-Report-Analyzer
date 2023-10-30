from dotenv import load_dotenv
from langchain.agents import create_csv_agent, tool
from langchain.chat_models import ChatOpenAI
import os

from regex import S

load_dotenv()

@tool
def csv_agent_tool(prompt: str) -> str:
    """Used to analyze CSV files."""
    agent = create_csv_agent(
        llm = ChatOpenAI(
        openai_api_key= os.getenv('OPENAI_API_KEY'),
        model="gpt-3.5-turbo",
        temperature=0,
    ), 

    max_iterations=5,
    verbose=True,
    )

    return agent(prompt)