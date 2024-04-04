import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_core.output_parsers import JsonOutputParser
import os
from tool.tools import compare_cumulative_returns_tool
from prompts.prompt_templates import cumulative_returns_page

load_dotenv()
MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

st.set_page_config(page_title='Cumulative Return Comparison',
                   page_icon='📊')
st.header('Compare Cumulative Returns 📊', divider='green')

st.markdown("This tool is used to compare the cumulative returns for the stock prices of multiple companies within a given date range.")

# session state
if "chat_history_cumulative" not in st.session_state:
    st.session_state.chat_history_cumulative = []

# conversation
for message in st.session_state.chat_history_cumulative:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history_cumulative.append(
        HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        template = cumulative_returns_page

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

        print("🟢", json_blob)

        start_date = json_blob.get("start")
        end_date = json_blob.get("end")
        tickers = [i for i in json_blob.get("tickers")]

        print(start_date)
        print(end_date)
        print(tickers)

        response = compare_cumulative_returns_tool(
            start_date, end_date, tickers)

        data = response["line"]

        try:
            df_data = {col: [x[i] for x in data['data']]
                       for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index("Date", inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

        st.write(response["comment"].replace("$", "\$"))

    st.session_state.chat_history_cumulative.append(
        AIMessage(content=response["comment"].replace("$", "\$")))
