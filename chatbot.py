pip install pypdf
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
import os

@st.cache_resource(show_spinner=False)
def load_documents(pdf_folder="docs"):
    financial_loader = PyPDFLoader(f"{pdf_folder}/financial_risk.pdf")
    operational_loader = PyPDFLoader(f"{pdf_folder}/operations_risk.pdf")
    financial_docs = financial_loader.load()
    operational_docs = operational_loader.load()
    return financial_docs + operational_docs

@st.cache_resource(show_spinner=False)
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

@st.cache_resource(show_spinner=True)
def create_vector_store(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

@st.cache_resource(show_spinner=True)
def load_llm():
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=256,
        do_sample=False,
        device=0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

@st.cache_resource(show_spinner=False)
def build_qa_chain(vectordb, llm):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain

def main():
    st.title("RBI Financial & Operational Risk Chatbot")

    # Load & prepare docs and model
    with st.spinner("Loading and processing documents..."):
        documents = load_documents()
        split_docs = split_documents(documents)
        vectordb = create_vector_store(split_docs)
        llm = load_llm()
        qa_chain = build_qa_chain(vectordb, llm)

    # Chat interface
    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Ask a question about RBI risk guidelines:")

    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)
            st.session_state.history.append((query, answer))

    if st.session_state.history:
        for i, (q, a) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

if __name__ == "__main__":
    main()
