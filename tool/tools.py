from dotenv import load_dotenv
from langchain.agents import create_csv_agent, tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os

load_dotenv()


@tool
def csv_agent_tool(prompt: str) -> str:
    """Used to analyze CSV files."""

    agent = create_csv_agent(
        path="combined_data.csv",
        llm=OpenAI(temperature=0),
        max_iterations=5,
        verbose=True,

    )

    return agent(prompt)
