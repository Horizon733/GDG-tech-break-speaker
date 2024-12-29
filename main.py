import os
from uuid import uuid1

import requests
import streamlit as st
from PIL import Image
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from constants import CHROMA_DB_DIR, AGRI_PROMPT_TEMPLATE
from utils import web_search
from utils import initialize_chromadb
import google.generativeai as genai


web_search_tool = Tool(
    name="WebSearch",
    func=lambda query: web_search(query),
    description="Search the web for relevant information."
)

# llm configurations
base_vector_store = initialize_chromadb(CHROMA_DB_DIR)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", timeout=None)
memory = ConversationBufferMemory(memory_key="history", return_messages=True, input_key="query")
prompt_template = PromptTemplate(input_variables=["context", "history", "query"], template=AGRI_PROMPT_TEMPLATE)
chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Streamlit app setup
st.set_page_config(page_title="Study Chat Assistant", layout="wide")
st.title("🌾 Farm help Assistant")
st.write("Ask questions about your soil, help with fertilizer selection, agriculture methods,and more!")

# User authentication (simplified)
user_id = st.text_input("User name", "")
if not user_id:
    st.warning("Please enter a User name to continue.")
    st.stop()

user_vector_store_id = f"{user_id}_{uuid1()}"

# Sidebar for uploading files
st.sidebar.title("Options")

uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)

    # Text box for user prompt
    prompt = st.sidebar.text_input("Enter your prompt related to the image:")
    send_btn_sidebar = st.sidebar.button("send", key="send_btn_sidebar")

    if send_btn_sidebar:
        # Process the image and prompt with Google Generative AI
        if prompt:
            try:
                # Save the image temporarily
                image_path = "tmp_images/temp_image.png"
                image.save(image_path)

                # Open the image from the saved path
                sample_file = Image.open(image_path)

                # Choose a Gemini model
                model = genai.GenerativeModel(model_name="gemini-1.5-pro")

                # Generate content based on the prompt and image
                response = model.generate_content([prompt, sample_file])

                # Display the response
                st.sidebar.write("Response from the vision model:")
                st.sidebar.write(response.text)
            except Exception as e:
                st.sidebar.error(f"Error processing the image and prompt: {e}")

# Chatting area
st.subheader("Chatbox")

chat_container = st.container()

def display_message(message, is_user=True):
    if is_user:
        chat_container.markdown(
            f"<div style='text-align: right; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>",
            unsafe_allow_html=True)
    else:
        chat_container.markdown(
            f"<div style='text-align: left; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>",
            unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with chat_container:
    for chat in st.session_state.messages:
        display_message(chat['content'], is_user=chat['is_user'])

user_input = st.text_input("Enter your query:", key="user_input")
send_button = st.button("Send")

if send_button:
    user_input = st.session_state.user_input.strip()
    if user_input:
        try:
            base_results = base_vector_store.similarity_search(user_input, k=3)
            context = ""
            for result in base_results:
                context += result.page_content + "\n\n"
        except Exception as e:
            st.error(f"Error retrieving context from ChromaDB: {e}")
            context = ""
        context += f"\nWeb search result:\n{web_search_tool.run(user_input)}\n"
        try:
            response = chain.run(context=context, query=user_input)
            st.session_state.messages.append({"role": "user", "content": user_input, "is_user": True})
            st.session_state.messages.append({"role": "assistant", "content": response, "is_user": False})
        except requests.exceptions.Timeout:
            st.error("The request to the LLM timed out. Please try again.")
        except Exception as e:
            st.error(f"Error generating response: {e}")
        st.rerun()
    else:
        st.warning("Please enter a valid query.")
