from fastapi import FastAPI
from src.embeddings import (
  embeddings,
  EmbeddingInput,
  EmbeddingOutput,
  list_models
)

app = FastAPI(
  title="Embeddings Benchmark"
)

@app.post("/embed", response_model=EmbeddingOutput)
def embed(data: EmbeddingInput):
  return embeddings(data)

@app.get("/list_models")
def list_models():
  return list_models



