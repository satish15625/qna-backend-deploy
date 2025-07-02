# RAG Pipeline API

A FastAPI-based Retrieval-Augmented Generation system for PDF document Q&A using ChromaDB and Ollama.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and start Ollama:**
   ```bash
   ollama pull mistral
   ollama serve
   ```

3. **Run the application:**
   ```bash
   python main.py
   # or
   uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Upload a PDF and ask questions:**
   ```bash
   # Upload document
   curl -X POST "http://localhost:8000/ingest" -F "file=@document.pdf"
   
   # Ask question
   curl -X POST "http://localhost:8000/ask-json" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'
   ```

## Documentation

- **[Complete API Documentation](API_DOCUMENTATION.md)** - Comprehensive guide with examples
- **[Function Reference](FUNCTION_REFERENCE.md)** - Quick lookup for developers

## Features

- Upload PDFs via file or URL
- Ask questions about document content
- Conversation history with caching
- Vector search with ChromaDB
- Local LLM inference with Ollama
