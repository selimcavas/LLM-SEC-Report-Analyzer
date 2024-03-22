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
from tool.tools import text2sql_tool

load_dotenv()
MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

st.set_page_config(page_title='Financial Data Analyzer',
                   page_icon='📉')
st.header('Financial Data Analyzer 📉', divider='green')

# session state
if "chat_history_sql" not in st.session_state:
    st.session_state.chat_history_sql = [
        AIMessage(
            content="This tool is designed to generate visual representations of a company's stock prices within a specified date range."),
    ]

# conversation
for message in st.session_state.chat_history_sql:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history_sql.append(
        HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        response = text2sql_tool(
            user_query)

        st.write(response.replace("$", "\$"))

    st.session_state.chat_history_sql.append(
        AIMessage(content=response.replace("$", "\$")))
