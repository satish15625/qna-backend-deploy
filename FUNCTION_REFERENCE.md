# Function Reference Guide

## API Endpoints

| Method | Endpoint | Parameters | Returns |
|--------|----------|------------|---------|
| `POST` | `/ingest` | `pdf_url?: HttpUrl, file?: UploadFile` | `{"status": "success message"}` |
| `POST` | `/ask-json` | `AskRequest: {question: str, sources?: list[str]}` | `{"answer": str}` |

## RAGPipeline Class

### Constructor
```python
RAGPipeline(embedding_model: str = "all-MiniLM-L6-v2", ollama_model: str = "mistral")
```

### Public Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `load_pdf` | `load_pdf(pdf_path: str) -> None` | Load and process PDF from filesystem |
| `load_pdf_from_url` | `load_pdf_from_url(pdf_url: str) -> None` | Download and process PDF from URL |
| `retrieve` | `retrieve(query: str, top_k: int = 3, sources: List[str] = None) -> List[str]` | Get relevant document chunks |
| `generate_answer` | `generate_answer(query: str) -> str` | Generate answer using RAG |
| `get_history` | `get_history() -> List[dict]` | Get Q&A history |

### Private Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `_load_history` | `_load_history() -> List[dict]` | Load history from JSON file |
| `_save_history` | `_save_history() -> None` | Save history to JSON file |
| `_chunk_pdf` | `_chunk_pdf(text: str, max_chunk_len: int = 500) -> List[str]` | Split text into chunks |
| `_generate_and_store_embeddings` | `_generate_and_store_embeddings(chunks: List[str]) -> None` | Create and store embeddings |

## Data Models

### AskRequest
```python
class AskRequest(BaseModel):
    question: str
    sources: Optional[list[str]] = None
```

## Quick Usage Examples

### Initialize RAG Pipeline
```python
from app.rag_pipeline import RAGPipeline
rag = RAGPipeline()
```

### Load Documents
```python
# From file
rag.load_pdf("./document.pdf")

# From URL
rag.load_pdf_from_url("https://example.com/doc.pdf")
```

### Query Documents
```python
# Basic query
answer = rag.generate_answer("What is the main topic?")

# With source filtering
chunks = rag.retrieve("search term", top_k=5, sources=["doc1.pdf"])
```

### API Calls
```bash
# Upload document
curl -X POST "http://localhost:8000/ingest" -F "file=@doc.pdf"

# Ask question
curl -X POST "http://localhost:8000/ask-json" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| Embedding Model | `all-MiniLM-L6-v2` | SentenceTransformer model |
| Ollama Model | `mistral` | Local LLM for generation |
| ChromaDB Path | `./chroma_store` | Vector database storage |
| History File | `./qa_history.json` | Q&A history storage |
| Server Host | `0.0.0.0` | FastAPI server host |
| Server Port | `8000` | FastAPI server port |

## Common Return Values

### Success Responses
- **Document Upload**: `{"status": "PDF content ingested [from URL/from file] successfully"}`
- **Question Answer**: `{"answer": "Generated answer based on documents"}`
- **Cached Answer**: `{"answer": "(From History) Previously generated answer"}`
- **No Context**: `{"answer": "❗ Answer not found in uploaded documents."}`

### Error Responses
- **HTTP 400**: `{"detail": "Error description"}`
- **No Input**: `{"detail": "No file or URL provided"}`
- **Download Fail**: `{"detail": "Failed to download PDF from URL"}`

## Performance Notes

- **Chunk Size**: 500 words max per chunk
- **Retrieval**: Top 3 relevant chunks by default
- **Response Time**: 1-3 seconds typical
- **Caching**: Instant response for repeated questions
- **Memory**: ~2-4GB RAM recommended