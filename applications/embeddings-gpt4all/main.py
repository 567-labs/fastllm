from fastapi import FastAPI
from src import InputRequest, get_embedding, OpenAIEmbeddingOutput

app = FastAPI()


@app.post("/v1/embedding", response_model=OpenAIEmbeddingOutput)
def process_embedding(data: InputRequest):
    return get_embedding(data)
