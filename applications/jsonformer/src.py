import json

from typing import Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class FunctionCall(BaseModel):
    name: str
    description: str
    parameters: Any


class Message(BaseModel):
    role: str
    content: str

    def __str__(self):
        return f"{self.role}: {self.content}"


class InputModel(BaseModel):
    model: str
    function_call: List[FunctionCall]
    messages: List[Message]


class FunctionCallResponse(BaseModel):
    name: str
    arguments: Optional[str] = None


class MessageResponse(BaseModel):
    role: str
    function_call: FunctionCallResponse


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


app = FastAPI()

# This should be moved to an env variable
MODEL = "databricks/dolly-v2-7b"

model = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def call_llm(messages: List[Message], json_schema: Any) -> str:
    prompt = "\n\n".join([str(message) for message in messages])
    prompt += "\n\n"
    prompt += "Make sure to follow the attribute descriptions and correctly return your data in the following format:\n\n"
    prompt += json.dumps(json_schema, indent=2)
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = json.dumps(jsonformer())
    return generated_data


def call_jsonformer(input_model: InputModel) -> OutputModel:
    # This is a simple mock function. Replace it with your actual logic.
    import time  # noqa: E402
    import uuid

    assert len(input_model.function_call) == 1, "Only one function call is supported"

    request_uuid = uuid.uuid4()

    function_name = input_model.function_call[0].name

    function_args = call_llm(
        messages=input_model.messages,
        json_schema=input_model.function_call[0].parameters,
    )

    response = OutputModel(
        id=f"chatcmpl-{request_uuid}",
        object="chat.completion",
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=MessageResponse(
                    role="assistant",
                    function_call=FunctionCallResponse(
                        name=function_name,
                        arguments=function_args,
                    ),
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    return response
