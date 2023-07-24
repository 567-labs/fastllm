from functools import lru_cache
import json
import time
import uuid

from typing import Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI

# This should be moved to an env variable
BREAK = "\n\n"
MODEL = "mosaicml/mpt-7b-instruct"


@lru_cache()
def model():
    import torch
    import transformers

    name = "mosaicml/mpt-7b-instruct"

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    config.attn_config["attn_impl"] = "triton"
    config.init_device = "cuda:0"  # For fast initialization directly on GPU!

    llm_model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    return llm_model, tokenizer


app = FastAPI(
    title="Jsonformer",
)


class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Message(BaseModel):
    role: str
    content: str

    def __str__(self):
        return f"{self.role}: {self.content}"


class InputModel(BaseModel):
    model: str = MODEL
    stream: bool = False
    functions: List[Function]
    messages: List[Message]


class FunctionCallResponse(BaseModel):
    name: str
    arguments: Optional[str] = None


class MessageResponse(BaseModel):
    role: str
    function_call: FunctionCallResponse
    content: Optional[str] = None


class Choice(BaseModel):
    index: int
    message: MessageResponse
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OutputModel(BaseModel):
    id: str
    object: str
    created: int
    choices: List[Choice]
    usage: Usage

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "function_call": {
                                "name": "extract",
                                "arguments": '{"name": "John Doe", "age": 42}',
                            },
                        },
                        "finish_reason": "function_call",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        }


def call_llm_with_jsonformer(messages: List[Message], function: Function) -> dict:
    from jsonformer import Jsonformer

    prompt = f"""
    ### Instruction:

    You will act as perfect extractor that extracts information out of the following messages in json.
    
    You have access to a schema `{function.name}`: `{function.description}`

    {json.dumps(function.parameters, indent=2)}

    {BREAK.join([str(message) for message in messages])}

    Returning `{function.name}` with the data according to the schema
    ### Response:
    """
    llm_model, tokenizer = model()

    jsonformer = Jsonformer(llm_model, tokenizer, function.parameters, prompt)
    return jsonformer()


def execute(input_model: InputModel) -> OutputModel:
    assert input_model.model == MODEL, "Only one model is supported"
    assert input_model.stream is False, "Only one stream is supported"
    assert len(input_model.functions) == 1, "Only one function call is supported"

    function_args = call_llm_with_jsonformer(
        messages=input_model.messages,
        function=input_model.functions[0],
    )

    response = OutputModel(
        id=f"{MODEL}-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=MessageResponse(
                    role="assistant",
                    function_call=FunctionCallResponse(
                        name=input_model.functions[0].name,
                        arguments=json.dumps(function_args),
                    ),
                ),
                finish_reason="function_call",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    return response


@app.post("/v1/chat/completions", response_model=OutputModel)
async def process_chat(input_model: InputModel):
    return execute(input_model)
