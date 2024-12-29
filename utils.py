import os
import tempfile

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from constants import LOCAL_OLLAMA_API_URL, AGRI_PROMPT_TEMPLATE


def initialize_chromadb(persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store


def web_search(query, max_results=5):
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for link in soup.find_all("a", class_="result__a", limit=max_results):
        results.append(link.get("href"))
    return results

