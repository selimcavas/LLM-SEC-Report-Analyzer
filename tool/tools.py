
import re
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import os
import json
from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from pinecone import Pinecone
# from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_models.fireworks import ChatFireworks

# after updates
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain.sql_database import SQLDatabase
from langchain import hub
from langchain.schema.output_parser import StrOutputParser

from data_models.models import StockPriceVisualizationToolParams, TranscriptAnalyzeToolParams, Text2SQLToolParams
from langchain_core.prompts import ChatPromptTemplate
# after new scract tool:
import sqlite3


load_dotenv()
# Bu toollar bir şekilde birden fazla paramater ile çağrılmalı ki böylece args_schema kullanımı anlamlı hale gelsin.

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"


@tool("transcript_analyze_tool", args_schema=TranscriptAnalyzeToolParams)
def transcript_analyze_tool(prompt: str) -> str:
    """
    Used to query data from a Pinecone index.

    """

    print('entered transcript tool')
    # Set the environment
    environment = "gcp-starter"

    index_name = "sec-filing-analyzer"

    # Create an instance of the Pinecone class
    pc = Pinecone(api_key=os.environ.get(
        "PINECONE_API_KEY"), environment=environment)

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Use the Pinecone instance to interact with the index
    # docsearch = pc.from_existing_index(index_name, embed)
    text_field = "text"

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name, embed, text_field)

    vectorstore.similarity_search(
        prompt,  # our search query
    )

    # Using LangChain we pass in our model for text generation.
    # llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=512)

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

    return str(qa(prompt))


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


# this is a new tool template ( not in production yet)
@tool("stock_prices_visualizer_tool", args_schema=StockPriceVisualizationToolParams)
def stock_prices_visualizer_tool(start_date: str, end_date: str, ticker: str, prompt: str) -> str:
    '''
    Used to visualize stock prices of a company in a given date range.
    '''
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

    output = ""
    # Prepare the output
    for row in rows:
        date, price = row
        output += f'{date}: {price}\n'

    chart_prompt = '''

        You are an experienced analyst that can generate stock price charts.
        Generate an appropriate chart for the stock prices of {ticker} between {start_date} and {end_date}.
        Use the {rows} and below output format for generating the chart for the question, do not round any values:
        
        {prompt} 

        The way you generate a chart is by creating a $JSON_BLOB.
        

        1. If the query requires a table, $JSON_BLOB should be like this:
           ```{{"table": 
                {{"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
            }}
           ```

        2. For a bar chart, $JSON_BLOB should be like this:
           ```{{"bar": 
                {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
           }}
           ```

        3. If a line chart is more appropriate, $JSON_BLOB should look like this:
           ```{{"line": 
                {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            }}
           ```

        Note: We only accommodate two types of charts: "bar" and "line".

        4. If the answer does not require a chart, simply respond with the following $JSON_BLOB:
           ```
            {{"answer": "Your answer here"}}
           ```
        
        ONLY return the $JSON_BLOB and nothing else. Do not write anything after the $JSON_BLOB.
        Do not write something like (Note: note), we only need the data. Make sure all opening and closing curly braces matches in the $JSON_BLOB.
        Begin!

        $JSON_BLOB:
        
    '''

    prompt_template = ChatPromptTemplate.from_template(chart_prompt)

    chat_model = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        }
    )

    # final_prompt = prompt_template.format(
    #     ticker=ticker, start_date=start_date, end_date=end_date, rows=output, prompt=prompt)

    response = prompt_template | chat_model | StrOutputParser()

    # Close the connection
    conn.close()

    return response.invoke({
        "ticker": ticker, "start_date": start_date, "end_date": end_date, "rows": output, "prompt": prompt
    })
