# NEXORA AI – Intelligent Personal Chatbot

## About the Project

NEXORA AI is an intelligent personal chatbot developed using Python, Streamlit, Ollama, and LangChain.

It allows users to interact with a locally running AI model, ask questions, receive intelligent responses, and maintain previous conversation history.

NEXORA provides different memory management options to improve conversational interactions.

## Features

- AI-powered chatbot interaction
- Local AI model integration using Ollama
- Conversation history management
- Create new conversations
- Multiple memory management options
- Conversation Buffer Memory
- Conversation Summary Memory
- Conversation Window Memory
- Support for multiple Ollama models
- Delete individual conversations
- Clear all chat history
- Local JSON-based chat storage
- Modern Midnight Blue and Teal user interface
- Dark theme design

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Streamlit | User interface |
| Ollama | Local AI model execution |
| LangChain | LLM integration and memory |
| JSON | Conversation storage |

## Supported AI Models

- Qwen 2.5 (3B)
- Gemma 3 (1B)
- Llama 3.2
- Mistral

**Note:** The models must be installed locally using Ollama.

## Screenshotss 

<img width="1841" height="892" alt="image" src="https://github.com/user-attachments/assets/8aa4d6f2-ab86-4b36-96b8-babd4a17ccda" />


## Memory Management

### 1. Conversation Buffer Memory

Stores the complete conversation history.

### 2. Conversation Summary Memory

Summarizes previous conversations to maintain important context.

### 3. Conversation Window Memory

Maintains the latest three conversation exchanges.

## Project Structure

    NEXORA AI/
    ├── app.py
    ├── requirements.txt
    ├── nexora_chats.json
    └── README.md

## Installation and Setup

### Step 1: Install Python

Download Python from:

https://www.python.org/downloads/

### Step 2: Install Ollama

Download Ollama from:

https://ollama.com/

### Step 3: Download the AI Model

    ollama pull qwen2.5:3b

### Step 4: Install Required Packages

    py -m pip install -r requirements.txt

### Step 5: Run the Application

    py -m streamlit run app.py

## Requirements

Create a `requirements.txt` file with the following packages:

    streamlit
    langchain-ollama
    langchain-classic
    langchain-core

## How It Works

1. Open the NEXORA AI application.
2. Select an available Ollama model.
3. Choose a memory management option.
4. Enter a question in the chat interface.
5. LangChain processes the conversation.
6. Ollama generates an AI response.
7. The response is displayed in the chatbot.
8. Conversation history is stored locally.

## Future Enhancements

- Voice input and output
- PDF upload and summarization
- Multilingual chatbot support
- AI-powered web search
- User authentication
- Chat history export

## Advantages

- Runs AI models locally.
- Maintains conversation history.
- Provides multiple memory options.
- Offers a simple and interactive interface.
- Stores conversations locally.
- Supports multiple AI models.

## Conclusion

NEXORA AI is a personal AI chatbot developed using Python, Streamlit, Ollama, and LangChain. It provides an interactive platform for communicating with locally running Large Language Models.

With conversation history and different memory management options, NEXORA AI improves the chatbot experience and demonstrates the practical application of Generative AI technologies.

## License

This project is developed for educational and learning purposes.
