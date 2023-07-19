from pydantic import BaseModel, Field
from typing import List
from gpt4all import Embed4All

# from functools import lru_cache

embedder = Embed4All()


class InputRequest(BaseModel):
    input: str = Field(..., description="The input text")


class Embedding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int = 0

    @classmethod
    def from_embedding(cls, embedding_list: List[float]):
        return cls(embedding=embedding_list, index=0)


class OpenAIEmbeddingOutput(BaseModel):
    object: str = "list"
    data: List[Embedding]

    class Config:
        schema_extra = {
            "example": {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.0023064255, -0.009327292],
                        "index": 0,
                    }
                ],
            }
        }


def get_embedding(data: InputRequest) -> OpenAIEmbeddingOutput:
    # Perform the embedding calculation here
    embedding = calculate_embedding(data.input)

    # Create the response
    embedding_obj = Embedding.from_embedding(embedding)
    return OpenAIEmbeddingOutput(data=[embedding_obj])


# @lru_cache(maxsize=128)
def calculate_embedding(text) -> List[float]:
    return embedder.embed(text)
