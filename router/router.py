from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from functions.functions import extract_zip, get_all_pdfs_from_folder, get_pdf_text, get_text_chunks, get_vector_store, process_user_input, log_to_json

router = FastAPI(debug=True)

@router.get("/")
async def home():
    return JSONResponse(content="Hello world!")

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a ZIP file containing PDFs.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")
    
    temp_dir = extract_zip(file.file)
    pdf_docs = get_all_pdfs_from_folder(temp_dir)
    if not pdf_docs:
        raise HTTPException(status_code=400, detail="No PDF files found in the ZIP.")
    
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    return {"status": "PDFs processed successfully"}

@router.post("/ask-question/")
async def ask_question(question: str):
    """
    Endpoint to ask a question based on the uploaded PDF context.
    """
    try:
        answer = process_user_input(question)
        log_to_json(model_name="model_name", question=question, answer=answer)
        return JSONResponse(content={"question": question, "answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

