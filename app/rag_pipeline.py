import os
import json
import time
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import ollama
import fitz  # PyMuPDF
import requests
import chromadb
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RAGPipeline:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", ollama_model: str = "mistral"):
        self.ollama_model = ollama_model
        self.embedder = SentenceTransformer(embedding_model)

        # ✅ Persistent vector store with Chroma
        self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_docs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        self.history_file = "./qa_history.json"
        self.history = self._load_history()

    def _load_history(self) -> List[dict]:
        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def _chunk_pdf(self, text: str, max_chunk_len: int = 500) -> List[str]:
        words = text.split()
        return [' '.join(words[i:i+max_chunk_len]) for i in range(0, len(words), max_chunk_len)]

    def _generate_and_store_embeddings(self, chunks: List[str]):
        embeddings = self.embedder.encode(chunks)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding.tolist()],
                ids=[f"doc_{len(self.collection.get()['ids']) + i}"]
            )

    def load_pdf(self, pdf_path: str):
        doc = fitz.open(pdf_path)
        all_text = "\n".join([page.get_text() for page in doc if page.get_text().strip()])
        chunks = self._chunk_pdf(all_text)
        source_name = os.path.basename(pdf_path)

        for i, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                ids=[f"{source_name}_{i}"],
                metadatas=[{"source": source_name}]
            )

    def load_pdf_from_url(self, pdf_url: str):
        response = requests.get(pdf_url)
        if response.status_code == 200:
            pdf_path = "./temp.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            self.load_pdf(pdf_path)
        else:
            raise Exception("Failed to download PDF from URL")

    def retrieve(self, query: str, top_k: int = 3, sources: List[str] = None) -> List[str]:
        try:
            query_vector = self.embedder.encode([query])
            all_docs = self.collection.get()
            if not all_docs['documents']:
                return []

            filtered_docs = [
                (doc, meta) for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])
                if sources is None or meta.get("source") in sources
            ]

            if not filtered_docs:
                return []

            docs, metas = zip(*filtered_docs)
            doc_vectors = self.embedder.encode(list(docs))
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [docs[i] for i in top_indices]

        except Exception:
            return []

    def generate_answer(self, query: str) -> str:
        start_time = time.time()

        # ✅ Return cached answer if already asked
        for item in self.history:
            if item['question'].strip().lower() == query.strip().lower():
                return f"(From History) {item['answer']}"

        contexts = self.retrieve(query)

        if not contexts:
            return "❗ Answer not found in uploaded documents."

        context_text = "\n".join(contexts)
        prompt = (
            "Answer the question strictly using the context below. "
            "If the answer is not in the context, respond with 'Answer not found in uploaded documents.'\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        )

        response = ollama.chat(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 128}  # Faster response
        )

        answer = response.get("message", {}).get("content", "").strip()
        if not answer:
            answer = "❗ Answer not found in uploaded documents."

        # ✅ Store with timestamp and save to disk
        qa_record = {
            "question": query,
            "answer": answer,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(qa_record)
        self._save_history()

        print(f"[INFO] Answer generated in {time.time() - start_time:.2f} seconds")
        return answer

    def get_history(self) -> List[dict]:
        return self.history
