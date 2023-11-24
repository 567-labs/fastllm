from fastapi import FastAPI
from src.main import (
    encode,
    EncodeInput,
    EncodeOutput,
    cosine_similarity,
    CosineSimilarityInput,
    CosineSimilarityOutput,
)

app = FastAPI(
    title="Sentence Transformer: Bi-Encoder",
)


@app.post("/encode", response_model=EncodeOutput)
def encode_request(input: EncodeInput):
    return encode(input)


@app.post("/cosine_similarity", response_model=CosineSimilarityOutput)
def cosine_similarity_request(input: CosineSimilarityInput):
    return cosine_similarity(input)
