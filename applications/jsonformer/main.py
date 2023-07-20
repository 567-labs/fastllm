from fastapi import FastAPI
from src import InputModel, OutputModel, call_jsonformer


app = FastAPI(
    title="Jsonformer",
    description="A FastAPI wrapper for the jsonformer library.",
    version="0.1.0",
)


@app.post("/v1/chat/completions", response_model=OutputModel)
async def process_chat(input_model: InputModel):
    return call_jsonformer(input_model)
