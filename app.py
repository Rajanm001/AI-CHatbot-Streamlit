import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Set up the Streamlit app title
st.title("future AI")

# Sidebar for app information
st.sidebar.header("About")
st.sidebar.write(
    "🌟 **future** is a context-aware AI chatbot application powered by Groq's API. 🤖✨\n\n"
    "It remembers the context of your conversation, making interactions more coherent and relevant. 🧠💬\n\n"
    "Select a model to start chatting. 🔑🛠️"
)

# Sidebar for settings
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Select a model:",
    [
        "gemma-7b-it",
        "gemma2-9b-it",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ],
    index=2
)

st.sidebar.warning(
    "Note: Each model has its own limits on requests and tokens. If you exceed these limits, you may encounter runtime errors."
)

# Set the API key directly in the code
API_KEY = "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"  # Hardcoded API key
os.environ["GROQ_API_KEY"] = API_KEY  # Store API key in environment variable

# Initialize session state for model and conversation history
if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = selected_model

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set up memory buffer for storing conversation history
memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

# Load previous chat history into memory
for message in st.session_state.chat_history:
    memory.save_context(
        {"input": message["human"]},
        {"output": message["AI"]}
    )

# Display previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and process it
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create a chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a helpful AI assistant named future."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ]
    )

    # Initialize Groq Chat with the selected model
    groq_chat = ChatGroq(
        groq_api_key=API_KEY,  # Use the hardcoded API key
        model_name=selected_model
    )

    # Initialize the conversation chain
    conversation = LLMChain(
        llm=groq_chat,
        prompt=chat_prompt,
        verbose=True,
        memory=memory
    )

    # Get the response from the AI model
    response = conversation.predict(human_input=prompt)

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save the message to session state for future reference
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append({"human": prompt, "AI": response})
