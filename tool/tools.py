
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

# after updates
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain.sql_database import SQLDatabase
from langchain import hub
from langchain.schema.output_parser import StrOutputParser

from data_models.models import StockPriceVisualizationToolParams, TranscriptAnalyzeToolParams, Text2SQLToolParams, CompareStockPriceVisualizationToolParams
from langchain_core.prompts import ChatPromptTemplate
# after new scract tool:
import sqlite3


load_dotenv()
# Bu toollar bir şekilde birden fazla paramater ile çağrılmalı ki böylece args_schema kullanımı anlamlı hale gelsin.

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"


@tool("transcript_analyze_tool", args_schema=TranscriptAnalyzeToolParams)
def transcript_analyze_tool(quarter: str, year: str, ticker: str) -> str:
    """
    Used to analyze earning call transcript texts and extract information that could potentially signal risks or growth within a company. Cannot be used with other tools.

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

    llm = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    analyze_prompt = '''
    As an expert financial analyst, analyze the earning call transcript texts and provide a comprehensive financial status of the company, indicating its growth or decline in value.
    
    Return a short $MARKDOWN_BLOB with the following information:

    ```
        - **Summary**: Short executive summary of the company's financial status, including key financial metrics.
        - **Analysis**: Short analysis of the company's financial performance, broken down by business segment if applicable.
        - **Key Points**: List of key points or themes from the earnings call, each with explanation, excerpts, and relevant data.
        - **Outlook**: Analysis of the company's future outlook, based on earnings call and financial data.
        - **Conclusion**: Conclusion that synthesizes the above information and highlights whether the company is on a growth trajectory.
    ```

    IMPORTANT: ONLY return the $MARKDOWN_BLOB and nothing else. $MARKDOWN_BLOB must be shorter than 100 words.
        Do not include any additional text, notes, or comments in your response. 
        Your response should begin and end with the $MARKDOWN_BLOB.
        Begin!

    $MARKDOWN_BLOB:

    '''

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            # search_type="mmr",
            search_kwargs={"k": 10, 'filter': {'source': f'{ticker.lower()}_{quarter.lower()}_{year}.txt'}}),
    )

    return str(qa(analyze_prompt))


@tool("text2sql_tool", args_schema=Text2SQLToolParams)
def text2sql_tool(text: str) -> str:
    """
    Used to convert user's prompts to SQL query to obtain financial data. Cannot be used with other tools.
    """
    chat_model = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
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
    Used to visualize stock prices of a company in a given date range. Cannot be used with other tools.
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

        You are an experienced analyst that can generate stock price charts and provide insightful comments about them.
        Generate an appropriate chart for the stock prices of {ticker} between {start_date} and {end_date}, and provide a brief comment about the price trends or significant events you notice in the data.
        Use the {rows} and below output format for generating the chart and the comment for the question, do not round any values:
        
        {prompt} 

        The way you generate a chart is by creating a $JSON_BLOB.
        

        1. If the query requires a table, $JSON_BLOB should be like this:
        ```{{"charttable": 
                {{"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}, "comment": "Your comment here"}}
            }}
        ```

        2. For a bar chart, $JSON_BLOB should be like this:
        ```{{"chartbar": 
                {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}, "comment": "Your comment here"}}
        }}
        ```

        3. If a line chart is more appropriate, $JSON_BLOB should look like this:
        ```{{"chartline": 
                {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}, "comment": "Your comment here"}}
            }}
        ```

        Note: We only accommodate two types of charts: "bar" and "line".

        4. If the answer does not require a chart, simply respond with the following $JSON_BLOB:
        ```
            {{"chartanswer": "Your answer here"}}
        ```
        
        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
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
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    # final_prompt = prompt_template.format(
    #     ticker=ticker, start_date=start_date, end_date=end_date, rows=output, prompt=prompt)

    response = prompt_template | chat_model | StrOutputParser()

    # Close the connection
    conn.close()

    return response.invoke({
        "ticker": ticker, "start_date": start_date, "end_date": end_date, "rows": output, "prompt": prompt
    })


@tool("compare_stock_prices_tool", args_schema=CompareStockPriceVisualizationToolParams)
def compare_stock_prices_tool(start_date: str, end_date: str, ticker1: str, ticker2: str, prompt: str) -> str:
    '''
    Used to compare stock prices and returns of two companies in a given date range. Cannot be used with other tools.
    '''
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Prepare the SQL queries
    sql_query1 = '''
        SELECT date, price
        FROM stock_prices
        WHERE date BETWEEN ? AND ? AND ticker = ?
        ORDER BY date
    '''
    sql_query2 = sql_query1  # The same query structure is used for the second ticker

    # Execute the SQL queries
    c.execute(sql_query1, (start_date, end_date, ticker1))
    rows1 = c.fetchall()
    c.execute(sql_query2, (start_date, end_date, ticker2))
    rows2 = c.fetchall()

    # Check if any data was fetched
    if not rows1 or not rows2:
        return f'No data for {ticker1} or {ticker2} between {start_date} and {end_date}'

    # Prepare the output and calculate cumulative returns
    first_price1 = rows1[0][1]
    output1 = [
        f'{date}: {price}, Cumulative Return: {((price - first_price1) / first_price1) - 1 }' for date, price in rows1]

    first_price2 = rows2[0][1]
    output2 = [
        f'{date}: {price}, Cumulative Return: {((price - first_price2) / first_price2) - 1}' for date, price in rows2]

    output1 = "\n".join(output1)
    output2 = "\n".join(output2)

    print("🟢", output1)
    print("🟢", output2)

    # Modify the chart_prompt to include instructions for comparing the two companies
    chart_prompt = '''
        As an experienced analyst, your task is to compare the cumulative returns of {ticker1} and {ticker2} between {start_date} and {end_date}. 

        Using the 
            Cumulative Return For {ticker1} : {output1}
            Cumulative Return For {ticker2} : {output2}

        You will need to generate a line graph with:
            - The x-axis representing the dates.
            - The y-axis representing the cumulative returns.
            - Two lines, one for each company, with the height of each point representing the cumulative return on that date.

        The graph should clearly show the comparative performance of the two companies over the given period. 

        Please include a brief analysis of the graph, highlighting any notable trends or points of interest.

        The way you generate a graph is by creating a $JSON_BLOB.

        For a line graph, $JSON_BLOB should be like this:
        ```{{"chartline": 
                {{"columns": ["Date", "{ticker1}", "{ticker2}"], "data": [["2020-01-01", value1, value2], ["2020-01-02", value1, value2], ...]}}, "comment": "Your comment here"}}
            }}

        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
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
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    response = prompt_template | chat_model | StrOutputParser()

    # Close the connection
    conn.close()

    return (response.invoke({
        "ticker1": ticker1, "ticker2": ticker2, "start_date": start_date, "end_date": end_date, "output1": output1, "output2": output2, "prompt": prompt
    }))
