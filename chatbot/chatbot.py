import warnings
from groq import Groq

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize the Groq client
groq_client = Groq(api_key="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")

# Function to get response from Groq models
def get_groq_response(user_input, chat_history):
    messages = chat_history + [
        {"role": "user", "content": user_input},
    ]

    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

# Chat function to interact with the user
def chat():
    print("Chatbot: Using Groq model.")

    chat_history = []
    print("Chatbot: Hello! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        bot_response = get_groq_response(user_input, chat_history)

        print("Chatbot:", bot_response)

        # Update chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": bot_response})

if __name__ == "__main__":
    chat()
