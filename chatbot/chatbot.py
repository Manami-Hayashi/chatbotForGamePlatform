from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from groq import Groq
from starlette.middleware.cors import CORSMiddleware
import warnings
import logging
from dotenv import load_dotenv
import os

# load env variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.ERROR)

# Initialize Groq client with environment variable for API key
# groq_api_key = "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"  # Replace with an environment variable in production
groq_client = Groq(api_key=GROQ_API_KEY)

# Define message model
class Message(BaseModel):
    role: str
    content: str

# Input model for chat endpoint
class ChatRequest(BaseModel):
    user_input: str
    chat_history: List[Message]



# Chatbot logic with Groq
def get_groq_response(user_input: str, chat_history: List[Message]) -> str:
    # Prepare messages
    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_input = request.user_input
        chat_history = request.chat_history

        # Get the response (from documents or chatbot)
        bot_response = get_groq_response(user_input, chat_history)
        return {"response": bot_response}
    except Exception as e:
        logging.error("Error in chat endpoint:", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
