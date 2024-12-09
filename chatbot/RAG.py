import os
import json
import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
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
                    page_content=f"Game: Checkers - {rule['title']}: {rule['description']}",
                    metadata={"type": "rule", "game": "Checkers", "id": rule.get("id")}
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

    for doc in doc_splits:
        if not hasattr(doc, "page_content") or not hasattr(doc, "metadata"):
            raise ValueError(f"Split document missing 'page_content' or 'metadata': {doc}")

    # Use OllamaEmbeddings as the embedding function
    try:
        embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
    except Exception as e:
        raise RuntimeError(f"Error initializing embedding function: {str(e)}")

    # Generate embeddings and store them in the vector store (Chroma)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embedding_function,
        collection_name="game_platform_rules",
        persist_directory="chroma_db" # it will be created automatically
    )
    retriever = vectorstore.as_retriever()

    # Determine if query is related to "Checkers rules" or "Platform guidance"
    if "checkers" in question.lower():
        retriever = vectorstore.as_retriever(filters={"game": "Checkers"})
    elif "platform" in question.lower():
        retriever = vectorstore.as_retriever(filters={"type": "platformInfo"})

    # initiialize the LLM model
    model_local = OllamaLLM(model="llama3.2:3b")

    after_rag_template = """You are an intelligent assistant with two purposes:\n
    1. Answering questions about game rules and breaking them into step-by-step instructions.\n
    2. Providing guidance about using the game platform.\n

    When asked about a game's rules, summarize the rules clearly or provide step-by-step instructions as requested.
    When asked about platform guidance, provide a direct and concise answer.

    Context:
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