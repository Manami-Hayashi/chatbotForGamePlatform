#!/bin/bash

# Check if Azure Key Vault name and secret name are set
if [[ -n "$AZURE_KEYVAULT_NAME" && -n "$AZURE_SECRET_NAME" ]]; then
  echo "Retrieving secret from Azure Key Vault..."

  # Authenticate with Azure (ensure the app has the necessary role assignments)
  az login --identity || exit 1

  # Retrieve the secret from Azure Key Vault
  az keyvault secret download \
    --vault-name "$AZURE_KEYVAULT_NAME" \
    --name "$AZURE_SECRET_NAME" \
    --file "/root/.ollama/id_ed25519" || exit 1

  chmod 600 /root/.ollama/id_ed25519
  echo "Secret retrieved and saved."
else
  echo "Azure Key Vault configuration is missing. Please check environment variables."
  exit 1
fi

# Start the FastAPI app
exec uvicorn chatbot:app --host 0.0.0.0 --port 8000
