
from dotenv import load_dotenv
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from tool.tools import csv_agent_tool, transcript_analyze_tool
import os

load_dotenv()


def run_main_agent(user_question):

    tool_list = []

    csv_tool = Tool(name="csv_tool", func=csv_agent_tool,
                    description='''Use this tool when user asks for a field value from a csv file or when asked for complex data analysis.
                    Whenever you decide to use this tool, do not use any other tools.
                    For complex data analysis, the tool will return the results in a tabular form with columns for 'Ticker', 'Field', and 'Value'.
                    When asked for complex data analysis such as finding the top 5 companies with the maximum income, present the results in a DataFrame with columns for 'Ticker', 'Field', and 'Value'.
                    Input should be a single string in JSON format. 
                    ''')

    transcript_tool = Tool(name="transcript_tool", func=transcript_analyze_tool,
                           description='''This tool is designed to handle queries related to earnings call transcripts. 
                           Whenever you decide to use this tool, do not use any other tools.
                           While trying to select the best Documents for the users question, extract ticker, quarter and year information from the question.
                           In the documents you choose make sure that the source key value in metadata is 'ticker_quarter_year.txt' and the year, quarter and ticker information is matching with the question.
                           If the ticker year or quarter of the question is not matching with source value in the document, do not use that document.
                           All of the document that you should use must have the same ticker, year and quarter information as the question.
                           You can have multiple documents that have the same source, while searching for the answer you have to iterate over multiple documents that have the same source tag.
                           When a user asks a question about an earnings call transcript, this tool will use the Pinecone vectors 
                           which hold the earnings call transcripts to find the relevant information.''')

    tool_list.append(csv_tool)
    tool_list.append(transcript_tool)

    prefix = """
    You are a financial data informant designed to chat with investors. 
    If the question is not related to financial data, please respond to user saying that you are unable to answer the question since you are only a chatbot.
    When user greets you explain what you can do briefly.
    Only use tools if it is necessary to answer the question, otherwise try to answer the question without using tools.
    You will be provided with a csv file called combined_data.csv regarding company's quarterly financial data from 10-Q filings. 
    The csv file has columns that represent the company ticker, field, and quarters. Each company has a unique ticker and they have individual rows for each field.
    Field name format is pascal case with spaces between words, however some fields have abbreviations, if you cannot find that field try to make all letters uppercase or use plural form of words, it can sometimes be mixed as well like 'Basic EPS'.
    Quarter column names are formatted like this year-quarter_number for example '2023-Q3'.
    If NaN values exists remove them before you do any mathematical calculation.
    Answer the following questions as best you can, and you have access to following tools.
    If you are asked to provide invesment idea or make suggestions, kindly respond to user saying that you are unable to make investment decisions since you are only a chatbot. 
    """

    suffix = """ Begin! 
        
        {chat_history}
        Question: {input} 
        {agent_scratchpad} """

    prompt = ZeroShotAgent.create_prompt(tool_list, prefix=prefix, suffix=suffix, input_variables=[
                                         "input", "chat_history", "agent_scratchpad"])
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(
        llm=ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0,
            verbose=True,
            model="gpt-3.5-turbo",
        ),
        prompt=prompt,
    )
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tool_list, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tool_list, verbose=True, memory=memory
    )

    return agent_chain.run(user_question)
