import tempfile
import os

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    return init_chat_model(model="command-r-plus-08-2024", temperature=0.2)


def process_pdf_into_vectorstore(uploaded_file, embedding_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    st.session_state.chunk_count = len(splits)

    vectorstore = FAISS.from_documents(splits, embedding_model)
    os.remove(temp_file_path)

    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
