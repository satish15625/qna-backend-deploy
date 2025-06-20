# ---- app/api.py ----
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, HttpUrl
from typing import Optional
from .rag_pipeline import RAGPipeline
import shutil
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAGPipeline using environment-configured models
rag_pipeline = RAGPipeline(
    embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    ollama_model=os.getenv("OLLAMA_MODEL", "mistral")
)

# Input schema for /ask-json
class AskRequest(BaseModel):
    question: str
    sources: Optional[list[str]] = None

# Ingest PDF via file or URL
@app.post("/ingest")
def ingest_pdf(
    pdf_url: Optional[HttpUrl] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        if pdf_url:
            rag_pipeline.load_pdf_from_url(str(pdf_url))
            return {"status": "✅ PDF content ingested from URL successfully"}

        elif file:
            file_location = "./temp_upload.pdf"
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            rag_pipeline.load_pdf(file_location)
            return {"status": "✅ PDF content ingested from file successfully"}

        else:
            raise HTTPException(status_code=400, detail="No file or URL provided")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Ask question and return answer using Ollama
@app.post("/ask-json")
def ask_question(request: AskRequest):
    try:
        answer = rag_pipeline.generate_answer(request.question, sources=request.sources)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
