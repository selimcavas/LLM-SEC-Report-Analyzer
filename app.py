import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len, 
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts= text_chunks,embedding= embeddings)
    return vectorstore

def main():
    load_dotenv()
    st.set_page_config(page_title='SEC Filing Analyzer', page_icon=':money_with_wings:')
    st.header('SEC Filing Analyzer :money_with_wings:')
    st.text_input('Ask a question about the SEC filings you provided:')

    with st.sidebar:
        st.subheader('Your SEC filings')
        pdf_docs = st.file_uploader('Upload your SEC filings here and click on Process', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store 
                vectorstore = get_vectorstore(text_chunks)
                


if __name__ == '__main__':
    main()