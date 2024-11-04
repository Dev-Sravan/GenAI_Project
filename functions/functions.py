import os
import json
import zipfile
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Tuple

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize constants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 10000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 1000))
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL = "models/embedding-001"

def extract_zip(zip_file: bytes) -> str:

    """Extract only PDF files from the uploaded ZIP file."""

    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file) as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith('.pdf'):
                zip_ref.extract(file_info, temp_dir)
    logger.info(f"Extracted PDFs to {temp_dir}")
    return temp_dir


def get_all_pdfs_from_folder(folder_path: str) -> List[str]:

    """Retrieve all PDF files from a folder."""

    pdf_docs = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.lower().endswith('.pdf')
    ]
    logger.info(f"Found {len(pdf_docs)} PDF files in {folder_path}")
    return pdf_docs


def get_pdf_text(pdf_docs: List[str]) -> List[Dict[str, str]]:

    """Extract text from a list of PDF files, preserving page numbers."""

    text_chunks = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_chunks.append({"text": page_text, "page_number": page_num})
    logger.info(f"Extracted text from {len(pdf_docs)} PDFs.")
    return text_chunks

def get_text_chunks(text_data: List[Dict[str, str]]) -> List[Dict[str, str]]:

    """Split the extracted text into manageable chunks, keeping page numbers."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks_with_page_numbers = []
    
    for item in text_data:
        page_number = item["page_number"]
        chunks = text_splitter.split_text(item["text"])
        for chunk in chunks:
            chunks_with_page_numbers.append({"text": chunk, "page_number": page_number})
    
    logger.info(f"Created {len(chunks_with_page_numbers)} text chunks.")
    
    return chunks_with_page_numbers

def get_vector_store(text_chunks: List[Dict[str, str]]) -> None:
    """Create a vector store using FAISS from text chunks with page metadata."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(
        [chunk["text"] for chunk in text_chunks],
        embedding=embeddings,
        metadatas=[{"page_number": chunk["page_number"]} for chunk in text_chunks]
    )
    vector_store.save_local("faiss_index")
    logger.info("Vector store created and saved.")

def get_conversational_chain() -> Tuple[str, any]:

    """Load a conversational chain for question-answering."""

    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, say, "answer is not available in the context".\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    llm_model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = create_stuff_documents_chain(llm_model, prompt)
    logger.info("Conversational chain created.")
    return chain

def process_user_input(user_question: str) -> Dict[str, any]:
    """Process user input and return the answer from the vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    database = FAISS.load_local(folder_path="faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search
    docs = database.similarity_search(user_question)
    logger.info(f"Retrieved {len(docs)} documents for the question.")

    context = ""
    page_numbers = set()
    
    for doc in docs:
        if 'page_number' in doc.metadata:
            page_numbers.add(doc.metadata['page_number'])
    page_numbers = tuple(sorted(page_numbers))
    chain = get_conversational_chain()
    response = chain.invoke({'context': docs, 'question': user_question})

    return {
        "question": user_question,
        "answer": response,
        "source": "from the pages "+ str(page_numbers) 
    }


def log_to_json(model_name: str = MODEL_NAME, question: str = "", answer: str = "", log_file: str = "qa_logs_1.json") -> None:
    """Log the user's question and answer to a JSON file."""
    log_entry = {
        model_name: {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r+') as file:
                logs = json.load(file)
                logs.append(log_entry)
                file.seek(0)
                json.dump(logs, file, indent=4)
        else:
            with open(log_file, 'w') as file:
                json.dump([log_entry], file, indent=4)
        logger.info(f"Logged Q&A to {log_file}.")
    except Exception as e:
        logger.error(f"Error logging Q&A: {e}")
