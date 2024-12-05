import os
import json
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Function to load and process the game rules JSON files
def load_all_data(files_list):
    all_data = []
    for file_name in files_list:
        file_path = os.path.join("C:\\Users\\mnmhy\\PycharmProjects\\chatbot\\files", file_name)
        try:
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    all_data.append(data)
            else:
                print(f"Unsupported file format: {file_name}")
        except FileNotFoundError:
            print(f"File {file_name} not found at {file_path}.")
    return all_data

# Function to extract the relevant content and create documents
def extract_content_and_create_documents(all_data):
    documents = []
    for data in all_data:
        if "rules" in data:
            for rule in data["rules"]:
                documents.append(Document(
                    page_content=f"{rule['title']}: {rule['description']}",
                    metadata={"type": "rule", "id": rule.get("id")}
                ))

        if "platformInfo" in data:
            platform_info = data["platformInfo"]
            documents.append(Document(
                page_content=f"Platform Info: {platform_info['description']}",
                metadata={"type": "platformInfo"}
            ))
            for key, value in platform_info["navigation"].items():
                documents.append(Document(
                    page_content=f"Navigation - {key}: {value['description']}",
                    metadata={"type": "navigation", "section": key}
                ))
                for feature in value["features"]:
                    documents.append(Document(
                        page_content=f"{key} - Feature: {feature}",
                        metadata={"type": "feature", "section": key}
                    ))
            if "quickTips" in platform_info:
                for tip, description in platform_info["quickTips"].items():
                    documents.append(Document(
                        page_content=f"Quick Tip - {tip}: {description}",
                        metadata={"type": "quickTip", "tip": tip}
                    ))
    print("Documents created:", documents)  # Debugging
    return documents

# Function to validate document format
def validate_documents(documents):
    for doc in documents:
        if not hasattr(doc, "page_content") or not hasattr(doc, "metadata"):
            raise ValueError(f"Invalid document format. Missing 'page_content' or 'metadata': {doc}")

# Function to process input question and answer generation
def process_input(question, documents):
    validate_documents(documents)

    # Split documents into smaller chunks using CharacterTextSplitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(documents)

    print("Document splits created:", doc_splits)  # Debugging
    for doc in doc_splits:
        if not hasattr(doc, "page_content") or not hasattr(doc, "metadata"):
            raise ValueError(f"Split document missing 'page_content' or 'metadata': {doc}")

    # Use OllamaEmbeddings as the embedding function
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # Generate embeddings and store them in the vector store (Chroma)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_function,
        collection_name="game_rules"
    )
    retriever = vectorstore.as_retriever()

    model_local = Ollama(model="llma3.2:3b")

    after_rag_template = """Answer the question based only on the following context:
       {context}
       Question: {question}
       """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

# Streamlit UI (for user interaction)
st.title("Game Platform Chatbot")
st.write("Enter a question related to game rules or platform usage:")

# Input fields for user question
question = st.text_input("Ask your question here:")

# Error handling and processing
if st.button('Query Game Rules'):
    with st.spinner('Processing...'):
        if question:
            try:
                # Load data and extract documents
                files_list = ["checkers_rules.json", "platform_guidance.json"]
                all_data = load_all_data(files_list)

                if not all_data:
                    st.error("No data loaded from JSON files.")
                else:
                    documents = extract_content_and_create_documents(all_data)

                    if not documents:
                        st.error("No documents created from the loaded data.")
                    else:
                        # Process input question
                        answer = process_input(question, documents)
                        st.text_area("Answer", value=answer, height=300, disabled=True)

            except Exception as e:
                st.error(f"Error processing the request: {str(e)}")
        else:
            st.error("Please enter a question.")
