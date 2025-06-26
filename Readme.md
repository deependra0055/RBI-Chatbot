# RBI Risk Guideline Chatbot using LangChain and RAG.

This project is a conversational chatbot built with LangChain, Hugging Face Transformers, and Streamlit, designed to answer questions related to Financial and Operational Risk Guidelines issued by the Reserve Bank of India (RBI).

It uses Retrieval-Augmented Generation (RAG) to ground its answers in official RBI documents, improving accuracy and transparency.

# Use Case
Helps users (analysts, students, regulators) navigate and understand RBI’s risk-related regulatory content by asking questions in plain English.

# Features

- Retrieval-Augmented Generation using LangChain
- Hugging Face `flan-t5` model for answer generation
- Embeds RBI PDFs using `sentence-transformers`
- Semantic search via ChromaDB vector store
- Streamlit web app for interactive chat

# Requirements
 Python 3.8+, langchain, chromadb, streamlit, transformers, sentence-transformers, protobuf==3.20.1
