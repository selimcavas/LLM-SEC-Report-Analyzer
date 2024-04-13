from curses import raw
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.fireworks import ChatFireworks
from langchain.schema.output_parser import StrOutputParser
from prompts.prompt_templates import prepare_report
import os
from fpdf import FPDF
import markdown

MODEL_ID = "accounts/fireworks/models/mixtral-8x7b-instruct"


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


prompt_template = ChatPromptTemplate.from_template(prepare_report)

chat_model = ChatFireworks(
    model=MODEL_ID,
    model_kwargs={
        "temperature": 0,
        "max_tokens": 2048,
        "top_p": 1,
    },
    fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
)

report = prompt_template | chat_model | StrOutputParser()

llm_report = report.invoke({
    "transcript_history": format_chat_history(transcript_history),
    "sql_history": format_chat_history(sql_history),
    "cumulative_history": format_chat_history(cumulative_history),
    "stock_compare_history": format_chat_history(stock_compare_history),
    "stock_prediction_history": format_chat_history(stock_prediction_history),
})

print(f'🟣Markdown Report:\n {llm_report}')

html_text = markdown.markdown(llm_report)

pdf = FPDF()
pdf.add_page()
pdf.set_font('helvetica', size=12)
pdf.write_html(html_text)
pdf.output("Analysis_Report.pdf")
