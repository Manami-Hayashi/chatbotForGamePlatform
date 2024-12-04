class GroqEmbeddings:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def generate_embeddings(self, text: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = json.dumps({"text": text})

        response = requests.post(self.endpoint, headers=headers, data=payload)

        if response.status_code == 200:
            return response.json()["embeddings"]  # Assuming the API returns embeddings in this field
        else:
            raise Exception(f"Error generating embeddings: {response.text}")