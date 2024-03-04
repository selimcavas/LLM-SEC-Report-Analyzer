from pyexpat import model
from tkinter.font import names
from xml.etree.ElementInclude import include
from annotated_types import doc
from dotenv import load_dotenv
from langchain.agents import tool, AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os
import json
from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA

from pinecone import Pinecone
#from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore

import pandas as pd
from langchain_community.chat_models.fireworks import ChatFireworks

## after updates
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain.sql_database import SQLDatabase
from langchain import hub
from langchain.schema.output_parser import StrOutputParser

from data_models.models import TranscriptAnalyzeToolParams, Text2SQLToolParams

## after new scract tool:
import yfinance as yf
import sqlite3
from datetime import datetime, timedelta


load_dotenv()
## Bu toollar bir şekilde birden fazla paramater ile çağrılmalı ki böylece args_schema kullanımı anlamlı hale gelsin.

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

@tool("transcript analyze", args_schema=TranscriptAnalyzeToolParams)
def transcript_analyze_tool(prompt: str) -> str:
    """
    Used to query data from a Pinecone index.
    
    """


    print('entered transcript tool')
    # Set the environment
    environment = "gcp-starter"

    index_name = "sec-filing-analyzer"
    
    # Create an instance of the Pinecone class
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"), environment=environment)

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Use the Pinecone instance to interact with the index
    #docsearch = pc.from_existing_index(index_name, embed)
    text_field = "text"

    vectorstore = PineconeVectorStore.from_existing_index(index_name, embed,text_field)

    vectorstore.similarity_search(
        prompt,  # our search query
    )

    # Using LangChain we pass in our model for text generation.
    #llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=512)
    
    llm = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        }
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 10}),  # return 7 most relevant docs
        # return_source_documents=True,
    )

    return qa(prompt)

@tool("text2sql_tool", args_schema=Text2SQLToolParams)
def text2sql_tool(text: str) -> str:
    """
    Used to convert user's prompts to SQL query to obtain financial data.
    """
    chat_model = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        }
    )

    few_shot = ""
    with open("sql_agent_prompts.json", "r") as file:
        few_shot = json.load(file)

    database = SQLDatabase.from_uri(database_uri="sqlite:///database.db")
    # check if connection is created successfully

    prompt = hub.pull("rlm/text-to-sql")

    # Create chain with LangChain Expression Language
    inputs = {
        "table_info": lambda x: database.get_table_info(),
        "input": lambda x: x["question"],
        "few_shot_examples": lambda x: few_shot,
        "dialect": lambda x: database.dialect,
    }

    sql_response = (
        inputs
        | prompt
        | chat_model.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    # Call with a given question
    response = sql_response.invoke(
        {"question": str(text)})

    print(response)

    start = response.find('"') + 1

    # Find the end of the SQL query
    end = response.rfind('"')

    # Extract the SQL query
    sql_query = response[start:end]
    print(sql_query)

    # Execute the generated SQL query on the database
    query_result = database._execute(sql_query)
    
    return query_result



## this is a new tool template ( not in production yet)
def stock_prices_tool(start_date: str, end_date: str, ticker: str) -> str:
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Prepare the SQL query
    sql_query = '''
        SELECT date, price
        FROM stock_prices
        WHERE date BETWEEN ? AND ? AND ticker = ?
        ORDER BY date
    '''

    # Execute the SQL query
    c.execute(sql_query, (start_date, end_date, ticker))
    rows = c.fetchall()

    # Check if any data was fetched
    if not rows:
        return f'No data for {ticker} between {start_date} and {end_date}'

    # Prepare the output
    output = f'Stock prices for {ticker} between {start_date} and {end_date}:\n'
    for row in rows:
        date, price = row
        output += f'{date}: {price}\n'

    # Close the connection
    conn.close()

    return output