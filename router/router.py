from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from functions.functions import *

router = FastAPI(debug=True)

# Allow CORS for front-end applications
router.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@router.get("/")
async def home():
    return JSONResponse(content={"message": "Hello world!"})

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload a ZIP file containing PDFs."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")
    
    temp_dir = extract_zip(file.file)
    pdf_docs = get_all_pdfs_from_folder(temp_dir)
    if not pdf_docs:
        raise HTTPException(status_code=400, detail="No PDF files found in the ZIP.")
    
    raw_text_data = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text_data)
    get_vector_store(text_chunks)

    return JSONResponse(content={"message": "PDFs processed successfully."})

@router.post("/ask/")
async def ask_question(user_question: str):
    """Endpoint to ask a question based on processed PDFs."""
    if not user_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    response = process_user_input(user_question)
    log_to_json("gemini-1.5-flash",user_question, response['answer'])
    
    return JSONResponse(content=response)
