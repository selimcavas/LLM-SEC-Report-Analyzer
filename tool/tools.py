
import re
from sre_parse import parse_template
from langchain_core.output_parsers import JsonOutputParser
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

from sympy import true

from data_models.models import StockPriceVisualizationToolParams, TranscriptAnalyzeToolParams, Text2SQLToolParams, CompareStockPriceVisualizationToolParams
from langchain_core.prompts import ChatPromptTemplate
# after new scract tool:
import sqlite3
from typing import List


load_dotenv()
# Bu toollar bir şekilde birden fazla paramater ile çağrılmalı ki böylece args_schema kullanımı anlamlı hale gelsin.

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"


def transcript_analyze_tool(quarter: str, year: str, ticker: str) -> str:
    """
    Used to analyze earning call transcript texts and extract information that could potentially signal risks or growth within a company.

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
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY"),
    )

    analyze_prompt = '''

        As an expert financial analyst, analyze the earning call transcript texts and provide a comprehensive financial status of the company, indicating its growth or decline in value.
        Prepare a markdown report that includes the following sections, each with the relevant information and data to support your analysis:
        
        ### Parameters:

        quarter: {quarter}
        year: {year}
        ticker: {ticker}

        ### Final Report Desired Format:

        - An executive summary of the company's financial status, including key financial metrics such as revenue, net income, and cash flow.
        - A detailed analysis of the company's financial performance, broken down by business segment if applicable.
        - A list of key points or themes from the earnings call, each with:
            - A brief explanation of why the point is important.
            - Relevant excerpts from the transcript, presented as bullet points, that illustrate or support the key point.
            - Any relevant financial data or metrics associated with the key point.
        - An analysis of the company's future outlook, based on statements made during the earnings call and the company's financial data.
        - A conclusion that synthesizes the above information and highlights whether the company is on a growth trajectory or facing a decline. This should include any significant risks or opportunities identified during the analysis.
        
        Final Report:
    '''
    prompt_template = ChatPromptTemplate.from_template(
        analyze_prompt)

    final_response = prompt_template.format_prompt(
        quarter=quarter, year=year, ticker=ticker).to_string()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            # search_type="mmr",
            search_kwargs={"k": 10, 'filter': {'source': f'{ticker.lower()}_{quarter.lower()}_{year}.txt'}}),
    )

    return qa(final_response)


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
        },
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    database = SQLDatabase.from_uri(database_uri="sqlite:///database.db")

    template = '''
        You are a financial data extractor. Analyze the following question.

        Question: {question}

        Table Info: {table_info}

        If the question pertains to financial data that can be extracted from the table, 
        return the word RELATED. If the question does not pertain to the table or the data in the table, 
        return the word UNRELATED. Do not add any additional information or comments.
        
        ***Only return RELATED or UNRELATED.***

        Begin!


        '''

    prompt_template = ChatPromptTemplate.from_template(template)

    response = prompt_template | chat_model | StrOutputParser()

    isRelated = response.invoke({
        "question": text, "table_info": database.get_table_info()
    })

    print("🟢", isRelated)

    if isRelated.strip() == "RELATED":

        few_shot = ""
        with open("sql_agent_prompts.json", "r") as file:
            few_shot = json.load(file)

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

        parse_template = ''' 
        Using the following sql query result and user question to form a short answer. Parse financial values using the seperator and return the result in a human readable format.
        
        User Question: {user_question}

        SQL Query Result: {query_result}

        Final Answer:
        '''

        parse_temp = ChatPromptTemplate.from_template(parse_template)

        response = parse_temp | chat_model | StrOutputParser()

        query_result = response.invoke(
            {"user_question": text, "query_result": query_result})

    elif isRelated.strip() == "UNRELATED":
        query_result = """Question is unrelated, please ask something related to the financial data available. Make sure you provide quarter and year information.
        For example:
        
        - Can you list top 5 companies based on the EBITDA data in 2023 q2?
        """
    else:
        query_result = "An error occured in the text2sql tool!"

    return query_result


def stock_prices_visualizer_tool(start: str, end: str, ticker: str) -> str:
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
    c.execute(sql_query, (start, end, ticker))
    rows = c.fetchall()

    chart_prompt = '''

        You are an experienced analyst that can generate stock price charts and provide insightful comments about them.
        Generate an appropriate chart for the stock prices of {ticker} between {start} and {end}, and provide a brief comment about the price trends or significant events you notice in the data.
        Use the {rows} and below output format for generating the $JSON_BLOB, do not round any values:
       

        $JSON_BLOB should look like this:
        ```{{"line": 
                {{"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}, "comment": "Your comment here"}}
            }}
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

    response = prompt_template | chat_model | JsonOutputParser()

    # Close the connection
    conn.close()

    return response.invoke({
        "ticker": ticker, "start": start, "end": end, "rows": rows
    })


def compare_cumulative_returns_tool(start: str, end: str, tickers: List[str]) -> str:
    '''
    Used to compare cumulative returns for the stock prices of multiple companies in a given date range. 
    '''
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Prepare the SQL queries
    sql_query = '''
        SELECT date, price
        FROM stock_prices
        WHERE date BETWEEN ? AND ? AND ticker = ?
        ORDER BY date
    '''

    # Execute the SQL query for each ticker and prepare the output
    outputs = []
    for ticker in tickers:
        c.execute(sql_query, (start, end, ticker))
        rows = c.fetchall()

        # Check if any data was fetched
        if not rows:
            return f'No data for {ticker} between {start} and {end}'

        # Prepare the output and calculate cumulative returns
        first_price = rows[0][1]
        output = [
            f'{date}: {((price - first_price) / first_price) * 100 }' for date, price in rows]
        output = "\n".join(output)
        outputs.append(output)

        # Print the outputs
    for i, output in enumerate(outputs):
        print(f"🟢 Output for {tickers[i]}: {output}")

    chart_prompt = '''
        As an experienced analyst, your task is to compare the cumulative returns of {tickers} between {start} and {end}. 

        Use the output from the SQL queries to display the cumulative returns for each company.
        SQL Output: {output}

        You will need to generate a line graph with:
            - The x-axis representing the dates.
            - The y-axis representing the cumulative returns.
            - A different line, for each company, with the height of each point representing the cumulative return on that date.

        The graph should clearly show the comparative performance of the the companies over the given period. 

        Please include a brief analysis of the graph, highlighting any notable trends or points of interest in the comment field.

        The way you generate a graph is by creating a $JSON_BLOB.

        $JSON_BLOB should be like this:
        ```{{"line": 
                {{"columns": ["Date", {tickers}], "data": [["2020-01-01", value1, value2, ...], ["2020-01-02", value1, value2 ...], ...]}}, "comment": "Your brief analysis and comparison here."}}
            }}
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

    response = prompt_template | chat_model | JsonOutputParser()

    # Close the connection
    conn.close()

    return response.invoke({
        "tickers": tickers, "start": start, "end": end, "output": outputs,
    })
