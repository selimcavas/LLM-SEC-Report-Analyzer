from tkinter.font import names
from xml.etree.ElementInclude import include
from dotenv import load_dotenv
from langchain.agents import tool, AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from llama_index.vector_stores import PineconeVectorStore
from langchain.chains import RetrievalQA
from torch import embedding
from langchain.vectorstores import Pinecone

load_dotenv()


@tool
def csv_agent_tool(prompt: str) -> str:
    """Used to analyze CSV files."""

    agent = create_csv_agent(
        path="combined_data.csv",
        llm=OpenAI(temperature=0),
        max_iterations=5,
        verbose=True,

    )

    return agent(prompt)


@tool
def transcript_analyze_tool(prompt: str) -> str:
    """Used to query data from a Pinecone index."""

    environment = "gcp-starter"

    api_key = os.environ['PINECONE_API_KEY']

    index_name = "sec-filing-analyzer"
    pinecone.init(api_key=api_key, environment=environment)

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    # This text field represents the field that the text contents of your document are stored in
    # text_field = "text"

    # load pinecone index for langchain
    # index = pinecone.Index(index_name)

    # vectorstore = Pinecone.from_existing_index(
    #    index_name, embed
    # )

    searcher = Pinecone.from_existing_index(
        index_name, embed
    )

    # retriever = vectorstore.as_retriever()

    # Query the vectorized data

    # Using LangChain we pass in our model for text generation.
    llm = ChatOpenAI(
        temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=512)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=searcher.as_retriever(),
        return_source_documents=True,
    )

    return qa(prompt)
