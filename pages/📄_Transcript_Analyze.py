from calendar import c
import json
from os import write
import re
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from streamlit_chat import message
from agent.agent import run_main_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_core.output_parsers import JsonOutputParser
import os
from st_chart_response import write_answer
from tool.tools import transcript_analyze_tool

load_dotenv()
MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

st.set_page_config(page_title='Earning Call Transcript Analyzer',
                   page_icon='📄')
st.header('Earning Call Transcript Analyzer 📄', divider='green')

# session state
if "chat_history_transcript" not in st.session_state:
    st.session_state.chat_history_transcript = [
        AIMessage(
            content="This tool is designed to analyze the text of earnings call transcripts and extract key information that could indicate potential risks or opportunities for growth within a company. "),
    ]

# conversation
for message in st.session_state.chat_history_transcript:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history_transcript.append(
        HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        template = '''
        You are an expert value extractor, look at the following question
         
          Question: {question} 
        
        Extract ticker symbol, year and quarter from the question. 
        If the question doesn't pertain to the earnings call transcript, please inform the user that you can't answer it. 
        Request the user to provide the ticker, year, and quarter information for the tool to function properly.

            ticker: Given ticker in the prompt taken from user (e.g. AAPL for Apple Inc)

            year: Given year in the prompt taken from user (e.g. 2023)

            quarter: Given quarter in the prompt taken from user in capital letters (e.g Q2)

        You should return a $JSON_BLOB with the extracted values such as: 

        ```
            {{
                "quarter": "Q2",
                "year": "2023",
                "ticker": "AAPL"
            }}
        ```

        IMPORTANT: ONLY return the $JSON_BLOB and nothing else. Do not include any additional text, notes, or comments in your response. 
        Make sure all opening and closing curly braces matches in the $JSON_BLOB. Your response should begin and end with the $JSON_BLOB.
        Begin!

        $JSON_BLOB:

        '''

        prompt_template = ChatPromptTemplate.from_template(template)

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

        json_blob = response.invoke({
            "question": user_query
        })

        print(json_blob)

        ticker = json_blob["ticker"]
        year = json_blob["year"]
        quarter = json_blob["quarter"]

        response = transcript_analyze_tool(
            quarter, year, ticker)

        st.write(response["result"].replace("$", "\$"))

    st.session_state.chat_history_transcript.append(
        AIMessage(content=response["result"].replace("$", "\$")))
