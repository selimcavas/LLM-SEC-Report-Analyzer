import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(
    page_title="Generate Report",
    page_icon="📝",
)

if "chat_history_transcript" not in st.session_state:
    transcript_history = []
else:
    transcript_history = st.session_state.chat_history_transcript

if "chat_history_sql" not in st.session_state:
    sql_history = []
else:
    sql_history = st.session_state.chat_history_sql

if "chat_history_cumulative" not in st.session_state:
    cumulative_history = []
else:
    cumulative_history = st.session_state.chat_history_cumulative

if "chat_history_stock_compare" not in st.session_state:
    stock_compare_history = []
else:
    stock_compare_history = st.session_state.chat_history_stock_compare

if "chat_history_stock_prediction" not in st.session_state:
    stock_prediction_history = []
else:
    stock_prediction_history = st.session_state.chat_history_stock_prediction


def format_chat_history(chat_history):
    history = []
    for message in chat_history:
        if isinstance(message, AIMessage):
            print("AI: " + message.content)
            history.append("AI: " + message.content)
        elif isinstance(message, HumanMessage):
            print("User: " + message.content)
            history.append("User: " + message.content)
        elif message is None:
            history.append("No chat history available.")
    return history


print(f'''Chat Histories:
      
    Transcript Analyze: {format_chat_history(transcript_history)}

    Financial Data Search: {format_chat_history(sql_history)}

    Cumulative Return Comparison: {format_chat_history(cumulative_history)}

    Stock Prices: {format_chat_history(stock_compare_history)}

    Stock Price Predictor: {format_chat_history(stock_prediction_history)}

      ''')
