from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Dict
import os
import logging
from dotenv import load_dotenv
from groq import Groq
from starlette.middleware.cors import CORSMiddleware
import warnings

# Suppress Groq-specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(dotenv_path="../chatbot.env")
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

# Chat endpoint
@app.post("/chat", response_model=Dict[str, str], summary="Chat Endpoint", description="Endpoint for chatbot interaction.")
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received request: {request.dict()}")  # Log the incoming request payload
    try:
        bot_response = get_groq_response(request.userInput, [m.dict() for m in request.chatHistory])
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
