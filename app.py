import json
from os import write
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from streamlit_chat import message
from agent.agent import run_main_agent
from langchain_core.messages import AIMessage, HumanMessage

from st_chart_response import write_answer


def main():
    load_dotenv()
    st.set_page_config(page_title='SEC Filing Analyzer',
                       page_icon=':money_with_wings:')
    st.header('SEC Filing Analyzer :money_with_wings:', divider='green')

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(
                content="Hello, I am a bot that can assist in financial data for NASDAQ100 companies. How can I help you?"),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = st.write_stream(run_main_agent(
                user_query, st.session_state.chat_history))

            json_candidate = response[1].get("messages")[0].content
            print("First JSON candidate: ", json_candidate)
            # Find the first and last curly brackets
            first_bracket = json_candidate.find('{')
            last_bracket = json_candidate.rfind('}')

            # Extract the substring between the first and last curly brackets
            json_candidate = json_candidate[first_bracket:last_bracket+1]
            print("Before loading JSON: ", json_candidate)

            json_blob = json.loads(json_candidate)
            print(json_blob)
            write_answer(json_blob)
            
            # Extract the comment from the JSON blob and write it to the app frontend
            comment = json_blob.get("comment")
            if comment:
                st.write(comment.replace("$", "\$"))

        st.session_state.chat_history.append(AIMessage(content=response))



if __name__ == '__main__':
    main()
