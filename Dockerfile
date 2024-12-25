# Stage 1: Ollama LLM Server
FROM ollama/ollama:latest as ollama-server

# Expose Ollama server's port for communication
EXPOSE 11434

# Start the Ollama server by default
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]


# Stage 2: FastAPI Application
FROM python:3.12-slim as chatbot-app

# Set the working directory in the container
WORKDIR /app

# Copy project files
COPY . /app

# Install required system packages and Python dependencies, and clean up intermediate files to reduce image size
RUN apt-get update && apt-get install -y gcc libpq-dev && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
# add a non-root user to run the application
RUN useradd -ms /bin/bash appuser
USER appuser

# Set environment variables for your application
ENV OLLAMA_KEY_PATH="/root/.ollama/id_ed25519"
ENV FILES_DIR="/app/files"
ENV VECTORSTORE_DIR="/app/chroma_db"

# Expose FastAPI port for user interaction
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
