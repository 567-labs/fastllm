import json
import time
import uuid

from typing import Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# This should be moved to an env variable
MODEL = "databricks/dolly-v2-7b"
BREAK = "\n\n"


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


app = FastAPI()


model = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def call_llm(messages: List[Message], function: Function) -> dict:
    prompt = f"""
    You will act as perfect router that executes functions or extracts json out of the following messages.
    
    You have access to a function `{function.name}`: `{function.description}`

    Arguments:
    {json.dumps(function.parameters, indent=2)}

    Messages:
    {BREAK.join([str(message) for message in messages])}

    Returning `{function.name}` with the following arguments:
    """

    jsonformer = Jsonformer(model, tokenizer, function.parameters, prompt)
    return jsonformer()


def call_jsonformer(input_model: InputModel) -> OutputModel:
    assert input_model.model == MODEL, "Only one model is supported"
    assert input_model.stream is False, "Only one stream is supported"
    assert len(input_model.functions) == 1, "Only one function call is supported"

    function_name = input_model.functions[0].name
    function_args = call_llm(
        messages=input_model.messages,
        function=input_model.functions[0],
    )

    response = OutputModel(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=MessageResponse(
                    role="assistant",
                    function_call=FunctionCallResponse(
                        name=function_name,
                        arguments=json.dumps(function_args),
                    ),
                ),
                finish_reason="function_call",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    return response
