import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from streamlit_chat import message

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain_core.output_parsers import JsonOutputParser
import os

from tool.tools import stock_prices_predictor_tool

load_dotenv()
MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"

st.set_page_config(page_title='Stock Price Predictor',
                   page_icon='🧠')
st.header('Stock Price Predictor 🧠', divider='green')

# session state
if "chat_history_stock_prediction" not in st.session_state:
    st.session_state.chat_history_stock_prediction = [
        AIMessage(
            content="This tool is used to predict stock prices of a company in a given date range."),
    ]

# conversation
for message in st.session_state.chat_history_stock_prediction:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history_stock_prediction.append(
        HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):

        template = '''
        You are an expert value extractor, look at the following question
         
          Question: {question} 
        
        Extract start date, end date and ticker from the question. 
        
            start: Start date for stock price visualization. In the format YYYY-MM-DD.
    
            end: End date for stock price visualization. In the format YYYY-MM-DD.
            
            ticker: Ticker for stock price visualization. For example, AAPL for Apple Inc.

        You should return a $JSON_BLOB with the extracted values such as: 

        ```
            {{
                "start": "2022-01-01",
                "end": "2022-01-01",
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

        start_date = json_blob.get("start")
        end_date = json_blob.get("end")
        ticker = json_blob.get("ticker")

        response = stock_prices_predictor_tool(
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

    st.session_state.chat_history_stock_prediction.append(
        AIMessage(content=response["comment"].replace("$", "\$")))
