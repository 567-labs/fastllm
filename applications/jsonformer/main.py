from typing import Any, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI


class FunctionCall(BaseModel):
    name: str
    description: str
    parameters: Any


class Message(BaseModel):
    role: str
    content: str


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


def call_jsonformer(input_model: InputModel) -> OutputModel:
    # This is a simple mock function. Replace it with your actual logic.
    import time  # noqa: E402
    import uuid

    request_uuid = uuid.uuid4()

    assert len(input_model.function_call) == 1, "Only one function call is supported"

    function_name = input_model.function_call[0].name

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
                        arguments="{}",
                    ),
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    return response


@app.post("/v1/chat/completions", response_model=OutputModel)
async def process_chat(input_model: InputModel):
    return call_jsonformer(input_model)
