
from dotenv import load_dotenv
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from tool.tools import csv_agent_tool, transcript_analyze_tool
from langchain.prompts import PromptTemplate
import os

load_dotenv()


def run_main_agent(user_question):

    csv_t = csv_agent_tool()

    tool_list = []

    csv_tool = Tool(name="csv_tool", func=csv_agent_tool,
                    description='''Use this tool when a user asks for a specific field value from a CSV file or requests complex data analysis. 
                        The tool has direct access to a DataFrame containing the company's quarterly financial data from 10-Q filings.
                        Do not change the field name user provides in the query, you can only change the capitilization of letters to find the exact matching.
                        The DataFrame is structured with 'Ticker', 'Field', and quarterly data columns. 
                        Ticker column contains the company's ticker, Field values are located in the 'Field' column, and the Quarter columns contain the data for the corresponding field.
                        For complex data analysis, such as finding the top 5 companies with the maximum income, the tool presents results in a DataFrame with 'Ticker', 'Field', and 'Value' columns.
                    ''')

    transcript_tool = Tool(name="transcript_tool", func=transcript_analyze_tool,
                           description='''This tool handles queries about earnings call transcripts by extracting ticker, quarter, and year details from the query, selecting matching documents from Pinecone vectors, and iterating over multiple documents with the same source tag to find the answer.''')

    tool_list.append(csv_tool)
    tool_list.append(transcript_tool)

    tools = [
        Tool.from_function(
            func=csv_t.run,
            name="csv_tool",
            description='''Use this tool when a user asks for a specific field value from a CSV file or requests complex data analysis. 
                        The tool has direct access to a DataFrame containing the company's quarterly financial data from 10-Q filings.
                        Do not change the field name user provides in the query, you can only change the capitilization of letters to find the exact matching.
                        The DataFrame is structured with 'Ticker', 'Field', and quarterly data columns. 
                        Ticker column contains the company's ticker, Field values are located in the 'Field' column, and the Quarter columns contain the data for the corresponding field.
                        For complex data analysis, such as finding the top 5 companies with the maximum income, the tool presents results in a DataFrame with 'Ticker', 'Field', and 'Value' columns.
                    ''',
        ),
        Tool.from_function(
            func=transcript_analyze_tool,
            name="transcript_tool",
            description='''This tool handles queries about earnings call transcripts by extracting ticker, quarter, and year details from the query, selecting matching documents from Pinecone vectors, and iterating over multiple documents with the same source tag to find the answer.''',
        ),
    ]

    prefix = '''As a chatbot, you provide financial data to investors. You can answer questions using tools when necessary and have access to a CSV file, 'combined_data.csv', containing companies' quarterly 10-Q filings. The CSV includes company tickers, fields, and quarters. Field names may vary in format, and 'NaN' values should be removed before calculations. 
    '''

    template = '''
    
     You can't provide investment advice or answer non-financial questions. If you're asked to provide investment advice, kindly reject the question by saying you are only a chatbot. If required; decide on a tool to use, and only use a single tool. Answer the following question:\n{question} using the tools only.
      
    '''

    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
        model="gpt-3.5-turbo",
        model_kwargs={"stop": ["\Observation:"]},
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        agent_kwargs={
            'prefix': prefix
        }
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=["question"])

    query_result = agent.run(
        prompt_template.format_prompt(question=user_question))

    return query_result
