# QnA Backend Deploy

FastAPI backend for PDF ingestion + RAG question answering.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

## Free deployment options

### 1) Render (free tier)
This repo already includes `render.yaml`.

- Build command: `pip install -r requirements.txt`
- Start command: `bash start.sh`
- Region/plan are set in `render.yaml`

**Important:** this project uses `ollama.chat(...)`, which needs an Ollama server available at runtime. On most free web tiers, you cannot run the full Ollama model server in the same small instance reliably.

Recommended approaches:
- Host Ollama separately (your own VM/local machine with public tunnel/VPN) and point backend to it.
- Or replace Ollama with a hosted LLM API for production deployment.

### 2) Railway (limited free trial credits)
- Create a new project from this repo.
- Use the same start command: `bash start.sh`.
- Add required env vars.

### 3) Fly.io (small free allowances may vary)
- Deploy using Dockerfile from this repo.
- Expose port `8000`.

## Required notes for successful deploy

1. **Entrypoint must be `app.api:app`** (not `app.main:app`).
2. Ensure persistent storage if you want to keep `chroma_store` and `qa_history.json` across restarts.
3. Configure CORS for your frontend domain in production.

## API endpoints

- `POST /ingest`
  - form-data: either `pdf_url` or `file`
- `POST /ask-json`
  - JSON body: `{ "question": "..." }`
