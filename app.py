import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from streamlit_chat import message
from agent.agent import run_main_agent
import os
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader
)
from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


def main():
    load_dotenv()
    st.set_page_config(page_title='SEC Filing Analyzer',
                       page_icon=':money_with_wings:')
    st.header('SEC Filing Analyzer :money_with_wings:', divider='green')

    user_question = st.chat_input('Ask a question about your CSV')

    if user_question is not None and user_question != "":
        with st.spinner(text="In progress..."):
            # response = agent.run(user_question)
            response = run_main_agent(user_question)
            if (
                "user_prompt_history" not in st.session_state
                and "chat_answer_history" not in st.session_state
            ):
                st.session_state["user_prompt_history"] = []
                st.session_state["chat_answer_history"] = []

            st.session_state["user_prompt_history"].append(user_question)
            st.session_state["chat_answer_history"].append(response)

            if st.session_state["chat_answer_history"]:
                for user_query, answer in zip(
                    st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]
                ):
                    message(user_query, is_user=True)
                    message(answer)


if __name__ == '__main__':
    main()
