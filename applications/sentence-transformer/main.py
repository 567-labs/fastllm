from functools import lru_cache
import uuid
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

app = FastAPI(
    title="Sentence Transformer",
)

# leaving this here so its easier to pick a model from the list
models_to_performance = {
    "ms-marco-TinyBERT-L-2-v2": 32.56,
    "ms-marco-MiniLM-L-2-v2": 34.85,
    "ms-marco-MiniLM-L-4-v2": 37.70,
    "ms-marco-MiniLM-L-6-v2": 39.01,
    "ms-marco-MiniLM-L-12-v2": 39.02,
}


class InputRequest(BaseModel):
    query: str
    docs: List[str]


class Document(BaseModel):
    text: str


class Result(BaseModel):
    score: float
    index: int
    document: Document


class OutputResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    results: List[Result]
    meta: dict = Field(default_factory=dict)


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODEL_NAME, max_length=512)


def score_data(input: InputRequest) -> OutputResponse:
    pairs = [[input.query, doc] for doc in input.docs]
    scores = model().predict(pairs)
    return OutputResponse(
        results=[
            Result(
                score=score,
                index=index,
                document=Document(text=doc),
            )
            for index, (score, doc) in enumerate(zip(scores, input.docs))
        ],
        meta={"model": MODEL_NAME},
    )


@app.post("/rerank", response_model=OutputResponse)
def rerank(data: InputRequest):
    return score_data(data)
