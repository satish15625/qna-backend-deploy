# RAG Pipeline API Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [API Endpoints](#api-endpoints)
3. [Core Components](#core-components)
4. [Data Models](#data-models)
5. [Usage Examples](#usage-examples)
6. [Setup and Configuration](#setup-and-configuration)
7. [Error Handling](#error-handling)

## Project Overview

This is a **Retrieval-Augmented Generation (RAG)** API built with FastAPI that allows users to:
- Upload PDF documents (via file upload or URL)
- Ask questions about the uploaded documents
- Get AI-generated answers based on document content
- Maintain conversation history with caching

### Technology Stack
- **FastAPI**: Web framework for building APIs
- **ChromaDB**: Vector database for document embeddings
- **SentenceTransformers**: For text embeddings (`all-MiniLM-L6-v2`)
- **Ollama**: Local LLM inference (`mistral` model)
- **PyMuPDF**: PDF text extraction
- **Uvicorn**: ASGI server

---

## API Endpoints

### Base URL
```
http://localhost:8000
```

### 1. Document Ingestion

#### `POST /ingest`

Ingests PDF documents into the RAG system for later querying.

**Request Parameters:**
- `pdf_url` (optional): HTTP URL of a PDF file
- `file` (optional): PDF file upload

**Request Examples:**

*Upload via URL:*
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "pdf_url=https://example.com/document.pdf"
```

*Upload via file:*
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@/path/to/document.pdf"
```

**Response:**
```json
{
  "status": "PDF content ingested from URL successfully"
}
```

**Error Responses:**
- `400 Bad Request`: No file or URL provided
- `400 Bad Request`: Failed to download PDF from URL
- `400 Bad Request`: Invalid PDF file

### 2. Question Answering

#### `POST /ask-json`

Ask questions about the ingested documents and receive AI-generated answers.

**Request Body:**
```json
{
  "question": "What is the main topic of the document?",
  "sources": ["document1.pdf", "document2.pdf"]  // optional
}
```

**Parameters:**
- `question` (required): The question to ask about the documents
- `sources` (optional): List of specific document sources to search in

**Request Example:**
```bash
curl -X POST "http://localhost:8000/ask-json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings in the research?",
    "sources": ["research_paper.pdf"]
  }'
```

**Response:**
```json
{
  "answer": "Based on the uploaded documents, the key findings include..."
}
```

**Special Response Cases:**
- If question was asked before: `"(From History) [cached answer]"`
- If no relevant context found: `"❗ Answer not found in uploaded documents."`

---

## Core Components

### RAGPipeline Class

The main class handling document processing and question answering.

#### Constructor

```python
RAGPipeline(embedding_model: str = "all-MiniLM-L6-v2", ollama_model: str = "mistral")
```

**Parameters:**
- `embedding_model`: SentenceTransformer model for creating embeddings
- `ollama_model`: Ollama model name for answer generation

**Example:**
```python
from app.rag_pipeline import RAGPipeline

# Initialize with default models
rag = RAGPipeline()

# Initialize with custom models
rag = RAGPipeline(
    embedding_model="all-mpnet-base-v2",
    ollama_model="llama2"
)
```

#### Public Methods

##### `load_pdf(pdf_path: str) -> None`

Loads and processes a PDF file from the local filesystem.

**Parameters:**
- `pdf_path`: Path to the PDF file

**Example:**
```python
rag.load_pdf("./documents/manual.pdf")
```

**Process:**
1. Extracts text from all PDF pages
2. Chunks text into manageable pieces (500 words max)
3. Stores chunks with embeddings in ChromaDB
4. Associates chunks with source document metadata

##### `load_pdf_from_url(pdf_url: str) -> None`

Downloads and processes a PDF from a URL.

**Parameters:**
- `pdf_url`: HTTP URL of the PDF file

**Example:**
```python
rag.load_pdf_from_url("https://example.com/document.pdf")
```

**Raises:**
- `Exception`: If PDF download fails

##### `retrieve(query: str, top_k: int = 3, sources: List[str] = None) -> List[str]`

Retrieves the most relevant document chunks for a given query.

**Parameters:**
- `query`: Search query
- `top_k`: Number of top results to return (default: 3)
- `sources`: Filter results by specific document sources

**Returns:**
- List of relevant text chunks

**Example:**
```python
# Get top 3 relevant chunks
chunks = rag.retrieve("machine learning algorithms")

# Get top 5 chunks from specific sources
chunks = rag.retrieve(
    "data preprocessing", 
    top_k=5, 
    sources=["ml_guide.pdf"]
)
```

##### `generate_answer(query: str) -> str`

Generates an answer to a question using retrieved document context.

**Parameters:**
- `query`: Question to answer

**Returns:**
- Generated answer string

**Example:**
```python
answer = rag.generate_answer("How does the algorithm work?")
print(answer)
```

**Features:**
- **Caching**: Returns cached answers for previously asked questions
- **Context-aware**: Uses relevant document chunks as context
- **History tracking**: Saves all Q&A pairs with timestamps
- **Fallback**: Returns error message if no relevant context found

##### `get_history() -> List[dict]`

Returns the complete question-answer history.

**Returns:**
- List of dictionaries with `question`, `answer`, and `timestamp`

**Example:**
```python
history = rag.get_history()
for item in history:
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}")
    print(f"Time: {item['timestamp']}")
    print("---")
```

#### Private Methods

##### `_load_history() -> List[dict]`
Loads Q&A history from JSON file.

##### `_save_history() -> None`
Saves current history to JSON file.

##### `_chunk_pdf(text: str, max_chunk_len: int = 500) -> List[str]`
Splits PDF text into chunks of specified maximum word length.

##### `_generate_and_store_embeddings(chunks: List[str]) -> None`
Creates embeddings for text chunks and stores them in ChromaDB.

---

## Data Models

### AskRequest

Pydantic model for question-answering requests.

```python
class AskRequest(BaseModel):
    question: str
    sources: Optional[list[str]] = None
```

**Fields:**
- `question`: The question string (required)
- `sources`: List of document sources to search (optional)

**Example:**
```python
request = AskRequest(
    question="What is the conclusion?",
    sources=["report.pdf", "summary.pdf"]
)
```

---

## Usage Examples

### Complete Workflow Example

```python
# 1. Initialize the RAG pipeline
from app.rag_pipeline import RAGPipeline
rag = RAGPipeline()

# 2. Load documents
rag.load_pdf("./research_paper.pdf")
rag.load_pdf_from_url("https://example.com/manual.pdf")

# 3. Ask questions
answer1 = rag.generate_answer("What is the main hypothesis?")
answer2 = rag.generate_answer("What are the limitations?")

# 4. View history
history = rag.get_history()
print(f"Asked {len(history)} questions so far")
```

### API Usage Examples

#### Upload Document and Ask Question

```bash
# 1. Upload a document
curl -X POST "http://localhost:8000/ingest" \
  -F "pdf_url=https://arxiv.org/pdf/2301.00001.pdf"

# 2. Ask a question
curl -X POST "http://localhost:8000/ask-json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the abstract of this paper?"
  }'
```

#### Multiple Document Workflow

```bash
# Upload multiple documents
curl -X POST "http://localhost:8000/ingest" -F "file=@doc1.pdf"
curl -X POST "http://localhost:8000/ingest" -F "file=@doc2.pdf"

# Ask question about specific source
curl -X POST "http://localhost:8000/ask-json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare the methodologies",
    "sources": ["doc1.pdf", "doc2.pdf"]
  }'
```

### Python Client Example

```python
import requests
import json

class RAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def upload_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/ingest", files=files)
        return response.json()
    
    def upload_pdf_url(self, pdf_url):
        data = {'pdf_url': pdf_url}
        response = requests.post(f"{self.base_url}/ingest", data=data)
        return response.json()
    
    def ask_question(self, question, sources=None):
        payload = {"question": question}
        if sources:
            payload["sources"] = sources
        
        response = requests.post(
            f"{self.base_url}/ask-json",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        return response.json()

# Usage
client = RAGClient()
client.upload_pdf("./document.pdf")
result = client.ask_question("What is the summary?")
print(result["answer"])
```

---

## Setup and Configuration

### Installation

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install Ollama:**
Follow [Ollama installation guide](https://ollama.ai) and pull the required model:
```bash
ollama pull mistral
```

3. **Run the Application:**
```bash
# Development mode
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# Or using the provided script
python main.py
```

### Environment Configuration

The application uses these configurable components:

- **Embedding Model**: `all-MiniLM-L6-v2` (stored in `./app/models/`)
- **Ollama Model**: `mistral` (must be available via Ollama)
- **Vector Store**: ChromaDB (persistent storage in `./chroma_store/`)
- **History Storage**: JSON file (`./qa_history.json`)

### File Structure

```
project/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── app/
│   ├── api.py             # FastAPI routes and app instance
│   ├── rag_pipeline.py    # Core RAG implementation
│   └── models/            # Pre-downloaded embedding models
├── chroma_store/          # ChromaDB vector storage (created on first run)
├── qa_history.json        # Q&A history (created on first run)
└── temp.pdf               # Temporary file for URL downloads
```

---

## Error Handling

### Common Error Scenarios

#### 1. Document Ingestion Errors

**No file or URL provided:**
```json
{
  "detail": "No file or URL provided"
}
```

**PDF download failure:**
```json
{
  "detail": "Failed to download PDF from URL"
}
```

**Invalid PDF file:**
```json
{
  "detail": "Error processing PDF: [specific error message]"
}
```

#### 2. Question Answering Errors

**General processing error:**
```json
{
  "detail": "Error generating answer: [specific error message]"
}
```

**No documents available:**
```json
{
  "answer": "❗ Answer not found in uploaded documents."
}
```

### Error Response Format

All API errors follow FastAPI's standard format:
```json
{
  "detail": "Error description"
}
```

### Troubleshooting

1. **Ollama Connection Issues:**
   - Ensure Ollama is running: `ollama serve`
   - Verify model is available: `ollama list`

2. **ChromaDB Issues:**
   - Check write permissions for `./chroma_store/` directory
   - Clear vector store if corrupted: `rm -rf ./chroma_store/`

3. **Memory Issues:**
   - Large PDFs may require more memory
   - Consider chunking large documents differently

4. **Model Loading Issues:**
   - Ensure `./app/models/all-MiniLM-L6-v2/` contains all required files
   - Re-download model if necessary

---

## Performance Considerations

- **Response Time**: Typical answer generation takes 1-3 seconds
- **Document Size**: Recommended maximum PDF size is 50MB
- **Concurrent Requests**: API supports multiple concurrent users
- **Memory Usage**: ~2-4GB RAM recommended for optimal performance
- **Caching**: Previously asked questions return instantly from cache

---

## Security Notes

- **CORS**: Currently configured to allow all origins (`"*"`)
- **File Upload**: No file type validation beyond PDF processing
- **URL Downloads**: No URL allowlist - can download from any HTTP(S) URL
- **Data Persistence**: All data stored locally (documents, embeddings, history)

For production deployment, consider:
- Implementing proper CORS policies
- Adding authentication/authorization
- File upload size limits
- URL allowlist for document downloads
- Data encryption for sensitive documents