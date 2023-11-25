from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
from pydantic import BaseModel, Field
from typing import List, Union

MODEL_NAME = "BAAI/bge-large-en-v1.5"


# Pydantic types
class EmbeddingInput(BaseModel):
    # modeled after request body from https://platform.openai.com/docs/api-reference/embeddings/create
    input: Union[str, List[str]] = Field(..., description="The input text(s) to embed")
    model: str = MODEL_NAME


class Embedding(BaseModel):
    # modeled after https://platform.openai.com/docs/api-reference/embeddings/object
    object: str = "embedding"
    embedding: List[float]
    index: int = 0


class Usage(BaseModel):
    # modeled after usage object from https://platform.openai.com/docs/api-reference/embeddings/create
    # TODO: implement token tracking
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingOutput(BaseModel):
    # modeled after response from https://platform.openai.com/docs/api-reference/embeddings/create
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: Usage = Field(..., default_factory=Usage)


class CosineSimilarityInput(BaseModel):
    text1: str
    text2: str


class CosineSimilarityOutput(BaseModel):
    similarity: float


# Functions


@lru_cache(maxsize=1)
def get_model(model_name=MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embeddings(req: EmbeddingInput) -> EmbeddingOutput:
    """Embed texts using the given sentence embedding model
    Source: https://www.sbert.net/docs/quickstart.html"""
    model = get_model(req.model)
    sentences = req.input if isinstance(req.input, List) else [req.input]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return EmbeddingOutput(
        data=[Embedding(embedding=e, index=i) for i, e in enumerate(embeddings)],
        model=req.model,
    )


def cosine_similarity(input: CosineSimilarityInput) -> CosineSimilarityOutput:
    """Calculate cosine similarity of 2 embedded strings using the given sentence embedding model
    Source: https://www.sbert.net/docs/quickstart.html"""
    model = get_model()
    emb1 = model.encode(input.text1)
    emb2 = model.encode(input.text2)
    cos_sim = util.cos_sim(emb1, emb2)
    # Convert tensor to float for pydantic output
    return CosineSimilarityOutput(similarity=cos_sim.tolist()[0][0])
