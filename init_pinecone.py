
from dotenv import load_dotenv
import os
import pinecone
from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print("initalize_pinecone2")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = "gcp-starter"

# Define the directory containing the text files
directory = Path('data_collection/transcripts_sample')

# Use a glob pattern to get all text files in the directory
txt_paths = directory.glob('*.txt')
# print('text paths: ', list(txt_paths))
# Initialize an empty list to store the documents
documents = []

# Iterate over the text file paths
for txt_path in txt_paths:
    # Open each text file and read its content
    with open(txt_path, 'r') as file:
        document_content = file.read()
    # Create metadata based on the file path
    metadata = {'source': txt_path.name}

    # Create a Document object with the content and metadata
    document = Document(page_content=document_content, metadata=metadata)

    # Add the document to the list
    documents.append(document)

# Set up the RecursiveCharacterTextSplitter, then split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# print('text', texts[0])

# PINECONE SETUP

index_name = 'sec-filing-analyzer'

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ.get("OPENAI_API_KEY"))

pc = PineconeVectorStore(
    index_name=index_name,
    pinecone_api_key=pinecone_api_key,
    embedding=embeddings,
    # namespace=pinecone_env
)

docsearch = pc.from_documents(texts, embeddings, index_name=index_name)

# Set the index name for this project in pinecone first

# index_name = 'sec-filing-analyzer'

# index_names = [index.name for index in pc.list_indexes()]

# print('index names: ', index_names)
# if index_name not in index_names:
#     pc.create_index(name=index_name, dimension=1536, metric="cosine")

# # Examine pinecone index. Delete all vectors, if you want to start fresh

# index = pc.Index(index_name)
# print(index.describe_index_stats())

# embeddings = OpenAIEmbeddings(
#     openai_api_key=os.environ.get("OPENAI_API_KEY"))

# # Create the vector store from the texts

# docsearch = pc.from_documents(
#     texts, embeddings, index_name=index_name)

# # for existing an vector store, use Pinecone.from_existing_index(index_name, embeddings)
