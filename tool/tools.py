from pyexpat import model
from tkinter.font import names
from xml.etree.ElementInclude import include
from annotated_types import doc
from dotenv import load_dotenv
from langchain.agents import tool, AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os
import pinecone
import json
from langchain.embeddings import OpenAIEmbeddings
from llama_index.vector_stores import PineconeVectorStore
from langchain.chains import RetrievalQA
from torch import embedding
from langchain.vectorstores import Pinecone
import pandas as pd

load_dotenv()


@tool
def csv_agent_tool(prompt: str) -> str:
    """Used to analyze CSV files."""

    csv_prompt = ""
    with open('prompts/csv_agent_prompts.json', 'r') as f:
        csv_prompt = json.load(f)

    df = pd.read_csv('combined_data.csv')

    print('entered csv agent')
    agent = create_pandas_dataframe_agent(
        df=df,
        llm=ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0,
            model="gpt-3.5-turbo",
            # model="gpt-4",
            model_kwargs={"stop": ["\Observation:"]},
        ),
        agent_type=AgentType.OPENAI_FUNCTIONS,  # fix here
        max_iterations=5,
        verbose=True,
        suffix=str(csv_prompt),
        include_df_in_prompt=None,
    )

    return agent.run(prompt)


def transcript_analyze_tool(prompt: str) -> str:
    """Used to query data from a Pinecone index."""

    print('entered transcript tool')

    environment = "gcp-starter"

    api_key = os.environ['PINECONE_API_KEY']

    index_name = "sec-filing-analyzer"
    pinecone.init(api_key=api_key, environment=environment)

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    docsearch = Pinecone.from_existing_index(
        index_name, embed
    )

    docsearch.similarity_search(
        prompt,  # our search query
    )

    # Using LangChain we pass in our model for text generation.
    llm = ChatOpenAI(
        temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=512)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(
            search_kwargs={"k": 7}),  # return 7 most relevant docs
        # return_source_documents=True,
    )

    return qa(prompt)


# if __name__ == "__main__":
#    response = csv_agent_tool(
#        "Can you list top 5 companies based on the EBITDA data in 2023 q2? Also get the ebitda values aswell")
#    print(response)
#
