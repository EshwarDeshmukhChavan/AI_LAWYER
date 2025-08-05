import asyncio
import streamlit as st
import os
import fitz  # PyMuPDF for fast PDF reading
from concurrent.futures import ThreadPoolExecutor
import hashlib

# --- LANGCHAIN IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("AI Legal Assistant for Indian Law")
st.write("Ask a question about the Indian Constitution or IPC based on the uploaded documents.")

# --- API KEY ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create a .streamlit/secrets.toml file with your GOOGLE_API_KEY.")
    st.stop()
except KeyError:
    st.error("GOOGLE_API_KEY not found in secrets. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

# --- FAST PDF READER WITH PROGRESS BAR ---
def load_pdf_fast(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    total_pages = len(doc)
    progress_bar = st.progress(0, text=f"Reading {os.path.basename(pdf_path)}...")
    
    for i, page in enumerate(doc):
        text += page.get_text()
        progress_bar.progress((i + 1) / total_pages, text=f"Reading {os.path.basename(pdf_path)} ({i+1}/{total_pages})")
    
    progress_bar.empty()  # Remove bar when done
    return text

# --- PROCESS SINGLE PDF ---
def process_pdf(pdf_file):
    pdf_path = os.path.join("data", pdf_file)
    try:
        text = load_pdf_fast(pdf_path)
        if not text.strip():
            st.warning(f"⚠️ No text found in {pdf_file} — it may be a scanned PDF.")
            return []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]
    except Exception as e:
        st.error(f"❌ Failed to process {pdf_file}: {e}")
        return []

# --- CACHE CHECK ---
def get_data_hash():
    """Generate a hash based on the PDFs to detect changes."""
    hash_md5 = hashlib.md5()
    for pdf_file in sorted(os.listdir("data")):
        if pdf_file.endswith(".pdf"):
            with open(os.path.join("data", pdf_file), "rb") as f:
                hash_md5.update(f.read())
    return hash_md5.hexdigest()

@st.cache_resource
def create_knowledge_base():
    data_dir = "data"
    if not os.path.exists(data_dir):
        st.error(f"The 'data' directory was not found. Please create it and add your PDF files.")
        st.stop()

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        st.warning("No PDF files found in the 'data' directory.")
        return None

    index_path = "faiss_index"
    current_hash = get_data_hash()

    # Load cached vector store if data hasn't changed
    if os.path.exists(index_path) and os.path.exists("data_hash.txt"):
        with open("data_hash.txt", "r") as f:
            old_hash = f.read().strip()
        if old_hash == current_hash:
            st.info("🔄 Loading cached knowledge base...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    st.info("📚 Processing PDFs in parallel...")
    all_chunks = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_files)
        for chunks in results:
            all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No text could be extracted from the PDFs.")
        return None

    st.info("⚡ Generating embeddings (batched)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(all_chunks, embedding=embeddings)

    st.success("✅ Knowledge base created successfully!")
    vector_store.save_local(index_path)
    with open("data_hash.txt", "w") as f:
        f.write(current_hash)

    return vector_store

knowledge_base = create_knowledge_base()

# --- QUESTION ANSWERING ---
def get_conversational_chain():
    prompt_template = """
    You are a helpful AI legal assistant. Your task is to answer the user's question based on the provided legal context.
    Provide a detailed and structured answer. If the context contains specific section numbers or articles, cite them.
    If the answer is not in the provided context, state that "the answer is not available in the provided documents."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

user_question = st.text_input("Ask your legal question:")

if user_question and knowledge_base:
    with st.spinner("Searching for the answer..."):
        docs = knowledge_base.similarity_search(user_question, k=3)
        chain = get_conversational_chain()
        response = chain.invoke({"input_documents": docs, "question": user_question})
        st.subheader("Answer:")
        st.write(response["output_text"])

elif user_question:
    st.warning("Knowledge base is not loaded. Cannot answer questions.")
