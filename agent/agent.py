
from dotenv import load_dotenv
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor, AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from tool.tools import transcript_analyze_tool, text2sql_tool,stock_prices_tool
from langchain.prompts import PromptTemplate
import os
from langchain_community.chat_models.fireworks import ChatFireworks

load_dotenv()


def run_main_agent(user_question):

    tool_list = [transcript_analyze_tool, text2sql_tool]

    prefix = '''As a chatbot, you provide financial data to investors. 
    You can answer questions using tools when necessary.
    '''

    template = '''
    
     You can't provide investment advice or answer non-financial questions. 
     If you're asked to provide investment advice, kindly reject the question by saying you are only a chatbot. 
     If required; decide on a tool to use, and only use a single tool. Answer the following question:\n{question} using the tools only.
      
    '''

    MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

    chat_model = ChatFireworks(
        model=MODEL_ID,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 1,
        }
    )

    agent = initialize_agent(
        tools=tool_list,
        llm=chat_model,
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

    query_result = query_result.replace("$", "\$")
    return query_result
