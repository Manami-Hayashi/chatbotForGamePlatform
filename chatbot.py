import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from starlette.middleware.cors import CORSMiddleware
from RAG import load_all_data, extract_content_and_create_documents, process_input

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine environment (local or Azure)
environment = os.getenv("ENVIRONMENT", "local")  # Default to "local"

# Configure URLs based on the environment
if environment == "azure":
    OLLAMA_URL = os.getenv("OLLAMA_URL", "https://ollama-app.azurecontainerapps.io")
else:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")  # Local Ollama

# Load file paths from environment variables
files_dir = os.getenv("FILES_DIR", "/files")
vectorstore_dir = os.getenv("VECTORSTORE_DIR", "chroma_db")

# Define message model
class Message(BaseModel):
    role: str
    content: str

# Input model for chat endpoint
class ChatRequest(BaseModel):
    userInput: str
    chatHistory: List[Message]

# Query Ollama service
def query_ollama(prompt: str) -> str:
    """Send a prompt to the Ollama service and retrieve the response."""
    import requests  # Only imported here to avoid unused imports locally
    try:
        headers = {"Content-Type": "application/json"}
        payload = {"model": "llama", "prompt": prompt}

        # Send the API request
        response = requests.post(f"{OLLAMA_URL}/v1/completions", json=payload, headers=headers)
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
        return {"response": bot_response}
    except ValueError as ve:
        logger.error("Validation error:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Error in chat endpoint:", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

# Read frontend URLs from environment variables
frontend_urls = os.getenv("FRONTEND_URLS", "http://localhost:5173").split(",")
backend_urls=os.getenv("BACKEND_URLS", "http://localhost:8090").split(",")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_urls,backend_urls],  # Adjust based on your frontend URL for better security
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
