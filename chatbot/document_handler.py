import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# load env variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class Document:
    def __init__(self, page_content: str):
        self.page_content = page_content

class DocumentHandler:
    def __init__(self):
        self.documents = []
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.vector_store = None
        self.embedding_model = None  # Initialize your embedding model here

    def load_document(self, file_path: str):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                documents = [Document(page_content=text)]
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
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(texts, self.embedding_model)

    def query_documents(self, query: str, k: int = 3):
        if self.vector_store is None:
            raise ValueError("No documents indexed. Load and index documents first.")
        results = self.vector_store.similarity_search(query, k=k)
        return [result.page_content for result in results]
