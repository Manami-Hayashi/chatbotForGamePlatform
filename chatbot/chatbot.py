from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from groq import Groq
from starlette.middleware.cors import CORSMiddleware
import warnings
import logging
from chatbot.document_handler import DocumentHandler
from dotenv import load_dotenv
import os

# load env variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI()

# Initialize DocumentHandler
document_handler = DocumentHandler()

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

# Load and index documents during startup
@app.on_event("startup")
async def load_documents():
    document_handler.load_document("checkers_rules.json")
    documents = document_handler.load_document("checkers_rules.json")
    document_handler.index_documents(documents)



# Chatbot logic with Groq
def get_groq_response(user_input: str, chat_history: List[Message]) -> str:
    # Check if the user query relates to game rules
    if "rules" in user_input.lower():
        try:
            document_results = document_handler.query_documents(user_input)
            return "Here are some relevant game rules:\n" + "\n".join(document_results)
        except Exception as e:
            return "Sorry, I couldn't find any relevant game rules."

    # Otherwise, proceed with the chatbot logic
    messages = chat_history + [{"role": "user", "content": user_input}]
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
