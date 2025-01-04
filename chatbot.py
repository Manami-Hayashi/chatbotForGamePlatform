from fastapi import FastAPI, HTTPException, Request, requests
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Dict
import os
import logging
from dotenv import load_dotenv
from groq import Groq
from starlette.middleware.cors import CORSMiddleware
import warnings
from RAG import load_all_data, extract_content_and_create_documents, process_input


# Suppress Groq-specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = "https://ollama-app.lemonwater-f19da583.westeurope.azurecontainerapps.io/"


# Load required environment variables
files_dir = os.getenv("FILES_DIR", "./files")
vectorstore_dir = os.getenv("VECTORSTORE_DIR", "chroma_db")

# Load environment variables from .env file
load_dotenv(dotenv_path="chatbot.env")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY environment variable is not set.")
    raise ValueError("GROQ_API_KEY is required for chatbot operation.")

# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Define message model
class Message(BaseModel):
    role: str
    content: str

# Input model for chat endpoint
class ChatRequest(BaseModel):
    userInput: str
    chatHistory: List[Message]

# Chatbot logic with Groq
def get_groq_response(userInput: str, chatHistory: List[Dict[str, str]]) -> str:
    messages = chatHistory + [{"role": "user", "content": userInput}]
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
        )
        logger.info(f"Groq response: {chat_completion}")
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error("Error generating Groq response", exc_info=True)
        return "An error occurred while processing your request."


# Query Ollama service
def query_ollama(prompt: str) -> str:
    """Send a prompt to the deployed Ollama service and retrieve the response."""
    try:
        ollama_url = "https://ollama-app.lemonwater-f19da583.westeurope.azurecontainerapps.io"  # Replace with your deployed app URL
        headers = {"Content-Type": "application/json"}
        payload = {"model": "llama", "prompt": prompt}

        # Send the API request
        response = requests.post(f"{ollama_url}/v1/completions", json=payload, headers=headers)
        response.raise_for_status()

        # Parse and return the response
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "No response from Ollama")
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        raise HTTPException(status_code=500, detail="Error querying Ollama service")


# Chat endpoint
@app.post("/chat", response_model=Dict[str, str], summary="Chat Endpoint", description="Endpoint for chatbot interaction.")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received request: {request.dict()}")  # Log the incoming request payload
    try:
        # Load data and extract documents
        files_list = ["checkers_rules.json", "platform_guidance.json"]
        all_data = load_all_data(files_list)
        documents = extract_content_and_create_documents(all_data)

        # Process input question using RAG
        bot_response = process_input(request.userInput, documents)
        # bot_response = get_groq_response(request.userInput, [m.dict() for m in request.chatHistory])
        return {"response": bot_response}
    except ValueError as ve:
        logger.error("Validation error:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Error in chat endpoint:", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for debugging
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred. Please try again later."})
