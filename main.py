# ---- main.py ----
import uvicorn

from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def index():
    return {"message": "Welcome to the RAG Pipeline API!"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
