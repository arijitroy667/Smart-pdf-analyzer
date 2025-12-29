# Pdf reader and explainer 
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
import concurrent.futures

import requests

load_dotenv()

#search tool
search_tool = DuckDuckGoSearchRun()

#llm
api_key = st.secrets["GOOGLE_API_KEY"]
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",api_key=api_key)

def pdf_text_extractor(file) -> str:
    """Extract text from a PDF file or file-like object."""

    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def text_splitter(text: str, chunk_size: int ,  overlap: int) -> list:
    """Split text into chunks of specified size with overlap."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def embeddings_vector_store(chunks: list) -> FAISS:
    """Generate embeddings for each text chunk and store in vector database."""
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

def retrieve_pdf_context(vector_store: FAISS, query: str, k: int = 4) -> str:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve_web_context(query: str) -> str:
    try:
        return search_tool.run(query)
    except Exception:
        return "No relevant web results found."

def retrieve_contexts(vector_store: FAISS, query: str):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pdf_future = executor.submit(retrieve_pdf_context, vector_store, query)
        web_future = executor.submit(retrieve_web_context, query)

        pdf_context = pdf_future.result()
        web_context = web_future.result()

    return pdf_context, web_context


# Main function to upload PDF and process
def pdf_upload_and_process(file_path: str)->  FAISS:
    """Upload PDF, extract text, split into chunks, generate embeddings and store in vector DB."""
    text = pdf_text_extractor(file_path)
    text_chunks = text_splitter(text, chunk_size=1000, overlap=200)
    documents = [ Document(page_content=chunk) for chunk in text_chunks ]
    vector_store = embeddings_vector_store(documents)
    return vector_store

#function to process query and retrieve contextual answer
def query_hybrid_rag(vector_store: FAISS, query: str) -> str:
    pdf_context, web_context = retrieve_contexts(vector_store, query)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer the question using:
1. PDF context (highest priority)
2. Web search results (only if PDF is insufficient)

Clearly separate insights from PDF and Web.

PDF Context:
{pdf_context}

Web Context:
{web_context}

Question:
{question}

Answer:
""",
        input_variables=["pdf_context", "web_context", "question"]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "pdf_context": pdf_context,
        "web_context": web_context,
        "question": query
    })

# vector_store = pdf_upload_and_process("Portfolio.pdf")
# query = "Give me the insights of various projects he/she has in thew portfolio."
# answer = query_hybrid_rag(vector_store, query)
# print("Answer:", answer)

# Streamlit UI
st.title("PDF Reader & Explainer")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = pdf_text_extractor(uploaded_file)
        text_chunks = text_splitter(text, chunk_size=1000, overlap=200)
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        vector_store = embeddings_vector_store(documents)
    st.success("PDF processed! You can now ask questions.")

    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Generating answer..."):
            answer = query_hybrid_rag(vector_store, query)
        st.markdown("**Answer:**")
        st.write(answer)
else:
    st.info("Please upload a PDF to get started.")