import streamlit as st
import os
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# --- LANGCHAIN IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import nest_asyncio
nest_asyncio.apply()  # Fix event loop issue

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("AI Legal Assistant for Indian Law")
st.write("Ask a question about the Indian Constitution or IPC based on the uploaded documents.")

# --- API KEY ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("GOOGLE_API_KEY not found in secrets. Please create .streamlit/secrets.toml with your key.")
    st.stop()

# --- Initialize models ---
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Google AI models. Please check your API key and network connection. Error: {e}")
    st.stop()

# --- PDF PROCESSING FUNCTIONS ---
def process_pdf_page(args):
    """Processes a single page, performing OCR if needed."""
    pdf_path, page_num, fitz_doc = args
    page_text = fitz_doc[page_num].get_text()
    if not page_text.strip():  # Likely scanned
        try:
            images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300)
            if images:
                page_text = pytesseract.image_to_string(images[0], lang="eng")
        except Exception as ocr_error:
            print(f"OCR failed on page {page_num} of {os.path.basename(pdf_path)}: {ocr_error}")
            return ""
    return page_text

def process_single_pdf(pdf_file):
    """Processes one full PDF."""
    pdf_path = os.path.join("data", pdf_file)
    try:
        with fitz.open(pdf_path) as doc:
            page_args = [(pdf_path, i, doc) for i in range(len(doc))]
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = executor.map(process_pdf_page, page_args)
                full_text = "\n".join(results)

        if not full_text.strip():
            print(f"⚠️ No text found in {pdf_file} even after OCR.")
            return []

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(full_text)
        return [Document(page_content=chunk, metadata={"source": pdf_file}) for chunk in chunks]

    except Exception as e:
        print(f"❌ Failed to process {pdf_file}: {e}")
        return []

# --- HASH FUNCTION ---
def get_data_hash():
    """Generate hash from PDFs to detect changes."""
    hash_md5 = hashlib.md5()
    data_dir = "data"
    if not os.path.exists(data_dir): return ""
    pdf_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pdf")])
    for pdf_file in pdf_files:
        with open(os.path.join(data_dir, pdf_file), "rb") as f:
            hash_md5.update(f.read())
    return hash_md5.hexdigest()

# --- LOAD OR CREATE KNOWLEDGE BASE ---
@st.cache_resource
def load_or_create_knowledge_base(_embeddings, force_refresh=False):
    data_dir = "data"
    index_path = "faiss_index"

    if not os.path.exists(data_dir):
        st.warning("No 'data' folder found. Please create it and add PDF files.")
        return None

    # Load existing index if available
    if not force_refresh and os.path.exists(index_path) and os.path.exists("data_hash.txt"):
        with open("data_hash.txt", "r") as f:
            old_hash = f.read().strip()
        if old_hash == get_data_hash():
            st.info("🔄 Loading existing knowledge base...")
            return FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)

    # Process PDFs
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        st.warning("No PDF files found in 'data'.")
        return None

    st.info(f"📚 Processing {len(pdf_files)} PDFs...")
    all_chunks = []
    with st.spinner("Extracting text..."):
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_single_pdf, pdf_files)
            for chunks in results:
                all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No text extracted from the PDFs.")
        return None

    st.info(f"⚡ Generating embeddings for {len(all_chunks)} chunks...")
    vector_store = FAISS.from_documents(all_chunks, embedding=_embeddings)
    vector_store.save_local(index_path)
    with open("data_hash.txt", "w") as f:
        f.write(get_data_hash())

    st.success("✅ Knowledge base ready!")
    return vector_store

# --- Force refresh button ---
force_refresh = st.button("♻️ Rebuild Knowledge Base")
knowledge_base = load_or_create_knowledge_base(embeddings, force_refresh)

# --- QA CHAIN ---
def get_conversational_chain(_llm):
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
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(_llm, chain_type="stuff", prompt=prompt)

# --- MAIN APP ---
user_question = st.text_input("Ask your legal question:")

if user_question and knowledge_base:
    with st.spinner("Searching for the answer..."):
        docs = knowledge_base.similarity_search(user_question, k=3)
        chain = get_conversational_chain(llm)
        response = chain.invoke({"input_documents": docs, "question": user_question})
        st.subheader("Answer:")
        st.write(response["output_text"])
elif user_question:
    st.warning("Knowledge base is not loaded. Cannot answer questions.")
