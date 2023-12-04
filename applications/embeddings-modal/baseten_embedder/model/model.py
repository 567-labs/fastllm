"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""


from sentence_transformers import SentenceTransformer
from functools import lru_cache
from pydantic import BaseModel, Field
from typing import List, Union

model_names = {
  "base": "BAAI/bge-base-en-v1.5",
  "small": "BAAI/bge-small-en-v1.5",
  "large": "BAAI/bge-large-en-v1.5",
}


# Declaring Pydantic Types
class EmbeddingInput(BaseModel):
  # modeled after request body from https://platform.openai.com/docs/api-reference/embeddings/create
  input: Union[str, List[str]] = Field(..., description="The input text(s) to embed")
  model: str=model_names["base"]

class Embedding(BaseModel):
  # modeled after embedding object from https://platform.openai.com/docs/api-reference/embeddings/object
  object: str = "embedding"
  embedding: List[float]
  index: int = 0  

class Usage(BaseModel):
  # modeled after usage object from https://platform.openai.com/docs/api-reference/embeddings/create
  # TODO implement token tracing later
  prompt_tokens: int = 0
  total_tokens: int = 0

class EmbeddingOutput(BaseModel):
  # modeled after embedding response from https://platform.openai.com/docs/api-reference/embeddings/create
  object: str
  data: List[Embedding]
  model: str
  usage: Usage

@lru_cache(maxsize=1)
def get_model(model_name=model_names["base"]) -> SentenceTransformer:
  return SentenceTransformer(model_name)

def embeddings(req):
  model = get_model(req["model"])
  sentences = req["input"] if isinstance(req["input"], List) else [req["input"]]
  embeddings = model.encode(sentences, convert_to_tensor=True)
  return EmbeddingOutput(
    object= "list",
    data=[Embedding(embedding=e, index=i) 
          for i, e in enumerate(embeddings)],
    model=req["model"],
    usage={ # TODO implement token tracing later
      "prompt_tokens": 0,
      "total_tokens": 0,
    }
  ).json()

class Model:
    def __init__(self, **kwargs) -> None:
        self._model = None

    def load(self):
        self._model = get_model

    def predict(self, model_input: EmbeddingInput) -> EmbeddingOutput:
        return embeddings(model_input)
