import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI, HuggingFaceHub
from streamlit_chat import message
from agent.agent import run_main_agent
import os
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    PromptHelper,
    load_index_from_storage,
    download_loader,
    VectorStoreIndex,
    SimpleDirectoryReader
)
from pathlib import Path


def initalize_pinecone():
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_env = "gcp-starter"
    index_name = "sec-filing-analyzer"

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=1536, metric="cosine")
    else:
        # If the index already exists, check if it is filled
        index = pinecone.Index(index_name)
        info = index.describe_index_stats()
        vector_number = info.namespaces[""].vector_count
        if vector_number == 4669:
            return None

    file_names = os.listdir("data_collection/transcripts")

    # Dictionary to store the indices
    indices_dict = {}

    for file_name in file_names:
        # Get the document ID by removing the file extension
        document_id = os.path.splitext(file_name)[0]
        # Use document_id as Pinecone title
        pinecone_title = document_id

        # Replace with appropriate metadata filters
        metadata_filters = {"name": document_id}
        vector_store = PineconeVectorStore(
            index_name=index_name,
            environment=pinecone_env,
            metadata_filters=metadata_filters
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)

    file_paths = Path('data_collection/transcripts')
    txt_paths = Path(file_paths).glob('*.txt')
    for txt_path in txt_paths:
        try:
            # In this particuler case we Load data froma directory containing pdf files, So we used the PDFReader loader.
            loader = SimpleDirectoryReader(input_files=[txt_path])
            aa_docs = loader.load_data()
            print(f"Loaded document from {txt_path}")
        except Exception as e:
            # Skips any files that may cause error while loading.
            print(
                f"Error reading Text file: {txt_path}. Skipping... Error:\n {e}")

        # Create the GPTVectorStoreIndex from the documents
        indices_dict[document_id] = GPTVectorStoreIndex.from_documents(
            aa_docs, storage_context=storage_context)
        indices_dict[document_id].index_struct.index_id = pinecone_title


def main():
    load_dotenv()
    initalize_pinecone()
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
