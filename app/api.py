# ---- app/api.py ----
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, HttpUrl
from typing import Optional
from .rag_pipeline import RAGPipeline
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for safety
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = RAGPipeline()

class AskRequest(BaseModel):
    question: str
    sources: Optional[list[str]] = None

@app.post("/ingest")
def ingest_pdf(
    pdf_url: Optional[HttpUrl] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        if pdf_url:
            result = rag_pipeline.load_pdf_from_url(str(pdf_url))
            return {"status": "ok", "ingestion": result}

        elif file:
            suffix = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file.file.read())
                file_location = tmp_file.name

            try:
                result = rag_pipeline.load_pdf(file_location)
            finally:
                if os.path.exists(file_location):
                    os.remove(file_location)

            return {"status": "ok", "ingestion": result}

        else:
            raise HTTPException(status_code=400, detail="No file or URL provided")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# @app.post("/ask-json")
# def ask_question(question: str = Form(...)):
#     try:
#         answer = rag_pipeline.generate_answer(question)
#         return {"answer": answer}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


# @app.post("/ask-json")
# def ask_question_json(request: AskRequest):
#     answer = rag_pipeline.generate_answer(request.question)
#     return {"answer": answer}


@app.post("/ask-json")
def ask_question(request: AskRequest):
    try:
        answer = rag_pipeline.generate_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))