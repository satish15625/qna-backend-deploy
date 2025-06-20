# File: app/main.py

from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from app import api  # api.py contains FastAPI instance

# app = api.app  # import the existing app

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API!"}

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
