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
from langchain.chains.Youtubeing import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="AI Legal Assistant", layout="wide")
st.title("🇮🇳 AI Legal Assistant for Indian Law")
st.write("Ask a question about the Indian Constitution or IPC based on the uploaded documents.")

# --- API KEY & MODEL INITIALIZATION (This is the stable structure) ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("GOOGLE_API_KEY not found in secrets. Please create .streamlit/secrets.toml with your key.")
    st.stop()

# --- Initialize models once to prevent async errors and for efficiency ---
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Google AI models. Please check your API key and network connection. Error: {e}")
    st.stop()

# --- PDF PROCESSING FUNCTIONS ---
def process_pdf_page(args):
    """Processes a single page, performing OCR if needed. Designed to be run in a thread."""
    pdf_path, page_num, fitz_doc = args
    page_text = fitz_doc[page_num].get_text()
    if not page_text.strip():  # If no text, it's likely a scanned image
        try:
            # Convert only the specific page to an image for OCR
            images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300)
            if images:
                page_text = pytesseract.image_to_string(images[0], lang="eng")
        except Exception as ocr_error:
            # This prints the error to your console/logs
            print(f"OCR failed on page {page_num} of {os.path.basename(pdf_path)}: {ocr_error}")
            return ""
    return page_text

def process_single_pdf(pdf_file):
    """Processes one full PDF file, with pages processed in parallel."""
    pdf_path = os.path.join("data", pdf_file)
    try:
        with fitz.open(pdf_path) as doc:
            # Prepare arguments for each page
            page_args = [(pdf_path, i, doc) for i in range(len(doc))]
            full_text = ""
            # Process pages in parallel for speed
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

def get_data_hash():
    """Generate a hash based on the PDFs to detect changes."""
    hash_md5 = hashlib.md5()
    data_dir = "data"
    if not os.path.exists(data_dir): return ""
    pdf_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pdf")])
    for pdf_file in pdf_files:
        with open(os.path.join(data_dir, pdf_file), "rb") as f:
            hash_md5.update(f.read())
    return hash_md5.hexdigest()

@st.cache_resource
def create_knowledge_base(_embeddings):
    data_dir = "data"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        st.warning("The 'data' directory is empty or not found. Please add PDF files.")
        return None

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        st.warning("No PDF files found in the 'data' directory.")
        return None

    index_path = "faiss_index"
    current_hash = get_data_hash()

    if os.path.exists(index_path) and os.path.exists("data_hash.txt"):
        with open("data_hash.txt", "r") as f:
            old_hash = f.read().strip()
        if old_hash == current_hash:
            st.info("🔄 Loading cached knowledge base...")
            return FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)

    st.info(f"📚 Found {len(pdf_files)} PDFs. Processing...")
    all_chunks = []
    with st.spinner("Processing documents... This may take a moment."):
        with ThreadPoolExecutor() as executor:
            # Process PDF files in parallel
            results = executor.map(process_single_pdf, pdf_files)
            for chunks in results:
                all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No text could be extracted from the PDFs. Ensure they are not scanned images.")
        return None

    st.info(f"⚡ Generating embeddings for {len(all_chunks)} text chunks...")
    vector_store = FAISS.from_documents(all_chunks, embedding=_embeddings)
    st.success("✅ Knowledge base created successfully!")
    vector_store.save_local(index_path)
    with open("data_hash.txt", "w") as f:
        f.write(current_hash)
    return vector_store

# Pass the initialized embeddings object into the function
knowledge_base = create_knowledge_base(embeddings)

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

# --- MAIN APP LOGIC ---
user_question = st.text_input("Ask your legal question:")

if user_question and knowledge_base:
    with st.spinner("Searching for the answer..."):
        docs = knowledge_base.similarity_search(user_question, k=3)
        chain = get_conversational_chain(llm) # Pass the initialized llm object
        response = chain.invoke({"input_documents": docs, "question": user_question})
        st.subheader("Answer:")
        st.write(response["output_text"])
elif user_question:
    st.warning("Knowledge base is not loaded. Cannot answer questions.")
