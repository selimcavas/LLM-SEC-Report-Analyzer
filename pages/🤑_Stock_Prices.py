import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_core.output_parsers import JsonOutputParser
import os
from tool.tools import stock_prices_visualizer_tool
from prompts.prompt_templates import stock_price_page

load_dotenv()
MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

st.set_page_config(page_title='Stock Price Analyzer & Visualizer',
                   page_icon='🤑')
st.header('Stock Price Analyzer & Visualizer 🤑', divider='green')

# session state
if "chat_history_stock_compare" not in st.session_state:
    st.session_state.chat_history_stock_compare = [
        AIMessage(
            content="This tool is used to visualize stock prices of a company in a given date range."),
    ]

# conversation
for message in st.session_state.chat_history_stock_compare:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history_stock_compare.append(
        HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        template = stock_price_page

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

        start_date = json_blob.get("start")
        end_date = json_blob.get("end")
        ticker = json_blob.get("ticker")

        response = stock_prices_visualizer_tool(
            start_date, end_date, ticker)

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

    st.session_state.chat_history_stock_compare.append(
        AIMessage(content=response["comment"].replace("$", "\$")))
