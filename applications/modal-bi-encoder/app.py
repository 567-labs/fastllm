from fastapi import FastAPI
from src.main import (
    embeddings,
    EmbeddingInput,
    EmbeddingOutput,
    cosine_similarity,
    CosineSimilarityInput,
    CosineSimilarityOutput,
)

app = FastAPI(
    title="Sentence Transformer: Bi-Encoder",
)


@app.post("/embeddings", response_model=EmbeddingOutput)
def embeddings_request(req: EmbeddingInput):
    """Create embedding vector(s) from input text(s).
    Modeled after OpenAI Create Embeddings endpoint <https://platform.openai.com/docs/api-reference/embeddings/create>
    """
    return embeddings(req)


@app.post("/cosine_similarity", response_model=CosineSimilarityOutput)
def cosine_similarity_request(input: CosineSimilarityInput):
    return cosine_similarity(input)
