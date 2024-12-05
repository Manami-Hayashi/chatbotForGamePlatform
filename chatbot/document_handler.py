import json
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, text
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv(dotenv_path="../chatbot.env")

class DocumentHandler:
    def __init__(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.vector_store = None

    def load_document(self, file_path: str):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents = [Document(page_content=file.read())]
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    documents = [Document(page_content=json.dumps(entry)) for entry in data]
                elif isinstance(data, dict):
                    documents = [Document(page_content=json.dumps(data))]
                else:
                    raise ValueError("Unsupported JSON data format.")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        return documents

    def index_documents(self, documents):
        """
        Split documents into chunks and create a vector index.
        """
        documents = [Document(page_content=doc.page_content) for doc in documents]
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)

    def query_documents(self, query: str, k: int = 3):
        """
        Query the vector store for similar documents.
        """
        if self.vector_store is None:
            raise ValueError("No documents indexed. Load and index documents first.")
        results = self.vector_store.similarity_search(query, k=k)
        return [result.page_content for result in results]
