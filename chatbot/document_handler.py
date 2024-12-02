from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class DocumentHandler:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.embedding_model = OpenAIEmbeddings()
        self.vector_store = None

    def load_document(self, file_path: str):
        # Example for PDF loading

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents = [{"text": text}]
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    documents = [{"text": json.dumps(entry)} for entry in data]
                elif isinstance(data, dict):
                    documents = [{"text": json.dumps(data)}]
                else:
                    raise ValueError("Unsupported JSON data format.")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        return documents

    def index_documents(self, documents):
        """
        Split documents into chunks and create a vector index.
        """
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(texts, self.embedding_model)

    def query_documents(self, query: str, k: int = 3):
        """
        Query the vector store for similar documents.
        """
        if self.vector_store is None:
            raise ValueError("No documents indexed. Load and index documents first.")
        results = self.vector_store.similarity_search(query, k=k)
        return [result.page_content for result in results]