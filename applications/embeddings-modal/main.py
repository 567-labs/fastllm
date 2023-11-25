from fastapi import FastAPI
from src.embeddings import (
  embeddings,
  EmbeddingInput,
  EmbeddingOutput,
)

app = FastAPI(
  title="Embeddings Benchmark"
)

@app.post("/embed", response_model=EmbeddingOutput)
def embed(data: EmbeddingInput):
  return embeddings(data)




