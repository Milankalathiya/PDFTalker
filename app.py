import os
import time

# load .env file before importing langchain
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag_pipeline import load_embedding_model, load_llm, process_pdf_into_vectorstore
from langchain_core.prompts import PromptTemplate


st.set_page_config(page_title="RAG Analytics Dashboard", page_icon="📊", layout="wide")

st.title("📊 RAG Analytics & Generation Matrix")

# Sidebar
with st.sidebar:
    st.header("📄 Document Setup")
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if st.button("Process PDF"):
        if pdf_file is not None:
            with st.spinner("Loading embedding model (first time may take a minute)..."):
                embedding_model = load_embedding_model()
            with st.spinner("Processing PDF..."):
                st.session_state.vectorstore = process_pdf_into_vectorstore(pdf_file, embedding_model)
                st.session_state.embedding_model = embedding_model
                st.success(f"✅ PDF processed! Total Chunks Created: {st.session_state.chunk_count}")
        else:
            st.warning("Please upload a file first.")

# Main Chat Interface
st.header("Chat Interface")
user_question = st.text_input("Ask a question about your PDF:", key="rag_user_question")

if user_question:
    if "vectorstore" not in st.session_state:
        st.error("Please upload and process a PDF first from the sidebar.")
    else:
        # PHASE 1: RETRIEVAL
        start_retrieval = time.time()

        retriever_engine = st.session_state.vectorstore
        docs_and_scores = retriever_engine.similarity_search_with_score(user_question, k=3)

        end_retrieval = time.time()
        retrieval_time = end_retrieval - start_retrieval

        avg_distance = sum(score for doc, score in docs_and_scores) / len(docs_and_scores) if docs_and_scores else 0
        total_words_retrieved = sum(len(doc.page_content.split()) for doc, score in docs_and_scores)

        # PHASE 2: GENERATION
        with st.spinner("LLM is generating answer..."):
            llm = load_llm()

            template = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer:"""
            prompt = PromptTemplate.from_template(template)

            context_text = "\n\n".join(doc.page_content for doc, score in docs_and_scores)
            formatted_prompt = prompt.format(context=context_text, question=user_question)

            start_generation = time.time()
            response = llm.invoke(formatted_prompt)
            end_generation = time.time()
            generation_time = end_generation - start_generation

            token_usage = response.response_metadata.get("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

        # PHASE 3: RENDER VISUAL MATRIX ON UI
        st.subheader("📈 System Metrics Matrix")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Retrieval Latency", value=f"{retrieval_time:.4f}s")
        with col2:
            st.metric(label="Avg Vector Distance", value=f"{avg_distance:.4f}", help="Lower scores mean tighter vector alignment.")
        with col3:
            st.metric(label="Generation Latency", value=f"{generation_time:.2f}s")
        with col4:
            st.metric(label="Tokens Generated", value=str(completion_tokens) if completion_tokens else "N/A")

        with st.expander("🔍 View Under-the-Hood Pipeline Trace", expanded=False):
            # 1. Trace Query Vector
            embedding_model = st.session_state.embedding_model
            query_vector = embedding_model.embed_query(user_question)
            st.markdown(f"**1. Query Embedded! (Showing first 10 of {len(query_vector)} dimensions)**")
            st.code(str(query_vector[:10]) + " ...")

            # 2. Trace Retrieved Text Segments
            st.markdown(f"**2. Top {len(docs_and_scores)} Chunks Fetched from FAISS Vector Store:**")
            for i, (doc, score) in enumerate(docs_and_scores):
                st.info(f"**Chunk {i+1} (Distance Score: {score:.4f}):**\n\n{doc.page_content}")

        st.markdown("---")
        st.write(f"**You:** {user_question}")
        st.success(f"**PDF Talker:** {response.content}")
