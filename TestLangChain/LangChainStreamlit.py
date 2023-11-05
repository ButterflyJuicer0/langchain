import os

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
directory = "PDF"


def load_file(directory):
    loader = DirectoryLoader(directory,glob="**/*.pdf")
    documents=loader.load()
    return documents


documents = load_file(directory)


def split_doc(documents,chunk_size=500,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

import pinecone
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)



