import os
import json
import zipfile
import tempfile
from datetime import datetime
from typing import List

from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize environment variables and constants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 10000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 1000))
MODEL_NAME = os.getenv("MODEL_NAME", "chat-model-001")
EMBEDDING_MODEL = "models/embedding-001"

def extract_zip(zip_file: bytes) -> str:
    """
    Extract only PDF files from the uploaded ZIP file.
    """
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file) as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.lower().endswith('.pdf'):
                zip_ref.extract(file_info, temp_dir)
    return temp_dir

def get_all_pdfs_from_folder(folder_path: str) -> List[str]:
    """
    Retrieve all PDF files from a folder.
    """
    pdf_docs = [os.path.join(root, file)
                for root, _, files in os.walk(folder_path)
                for file in files if file.lower().endswith('.pdf')]
    return pdf_docs

def get_pdf_text(pdf_docs: List[str]) -> str:
    """
    Extract text from a list of PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text: str) -> List[str]:
    """
    Split the extracted text into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks: List[str]) -> None:
    """
    Create a vector store using FAISS from text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain() -> tuple:
    """
    Load a conversational chain for question-answering.
    """
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
    return MODEL_NAME, chain

def process_user_input(user_question: str) -> str:
    """
    Process the user's question and retrieve an answer using the stored vector database.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    database = FAISS.load_local(folder_path="faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = database.similarity_search(user_question)

    _, chain = get_conversational_chain()
    response = chain.invoke({'context': docs, 'question': user_question})
    return response

def log_to_json(model_name: str, question: str, answer: str, log_file="qa_logs_1.json") -> None:
    """
    Log the user's question and answer to a JSON file.
    """
    log_entry = {
        model_name: {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
    }
    if os.path.exists(log_file):
        with open(log_file, 'r+') as file:
            logs = json.load(file)
            logs.append(log_entry)
            file.seek(0)
            json.dump(logs, file, indent=4)
    else:
        with open(log_file, 'w') as file:
            json.dump([log_entry], file, indent=4)
