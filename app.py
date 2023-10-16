import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI


def main():
    load_dotenv()
    st.set_page_config(page_title='SEC Filing Analyzer', page_icon=':money_with_wings:')
    st.header('SEC Filing Analyzer :money_with_wings:')

    user_csv = st.file_uploader('Upload a CSV file', type='csv')
    
    if user_csv is not None: 
        user_question = st.text_input('Ask a question about your CSV') 
        llm = OpenAI()
        agent = create_csv_agent(llm, user_csv, verbose=True)
        

        if user_question  is not None and user_question != "":
            with st.spinner(text="In progress..."):
                response = agent.run(user_question)
                st.write(response)

if __name__ == '__main__':
    main()