# Stage 1: Ollama LLM Server
FROM ollama/ollama:latest AS ollama-server

# Expose Ollama server's port for communication
EXPOSE 11434

# Start the Ollama server by default
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]

# Stage 2: FastAPI Application
FROM python:3.12-slim AS chatbot-app

# Set the working directory in the container
WORKDIR /app

# Copy project files
COPY . /app

# Install required system packages and Python dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev curl && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Switch to root user to adjust file permissions
USER root
# Copy and modify permissions for startup script
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Add a non-root user for running the application
RUN useradd -ms /bin/bash appuser
USER appuser

# Set environment variables for your application
ENV FILES_DIR="/app/files"
ENV VECTORSTORE_DIR="/app/chroma_db"

# Expose FastAPI port for user interaction
EXPOSE 3000

# Command to run the FastAPI app
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "3000"]
