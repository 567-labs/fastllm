from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
from pydantic import BaseModel
from typing import List

MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Pydantic types


class EncodeInput(BaseModel):
    texts: List[str]


class EncodeOutput(BaseModel):
    embeddings: List[List[float]]


class CosineSimilarityInput(BaseModel):
    text1: str
    text2: str


class CosineSimilarityOutput(BaseModel):
    similarity: float


# Functions


@lru_cache(maxsize=1)
def get_model(model_name=MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode(input: EncodeInput) -> EncodeOutput:
    """Encode a List[str] using the given sentence embedding model
    Source: https://www.sbert.net/docs/quickstart.html"""
    model = get_model()
    embeddings = model.encode(input.texts, convert_to_tensor=True)
    # Conver tensor to a List[float] for pydantic output
    return EncodeOutput(embeddings=embeddings.tolist())


def cosine_similarity(input: CosineSimilarityInput) -> CosineSimilarityOutput:
    """Calculate cosine similarity of 2 embedded strings using the given sentence embedding model
    Source: https://www.sbert.net/docs/quickstart.html"""
    model = get_model()
    emb1 = model.encode(input.text1)
    emb2 = model.encode(input.text2)
    cos_sim = util.cos_sim(emb1, emb2)
    # Convert tensor to float for pydantic output
    return CosineSimilarityOutput(similarity=cos_sim.tolist()[0][0])
