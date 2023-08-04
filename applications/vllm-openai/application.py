# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py
from config import download_model_to_folder, MODEL
from modal import Image, Secret, Stub, method, asgi_app
from http import HTTPStatus
import time
import logging
from typing import AsyncGenerator, Optional
from packaging import version

import fastapi
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from protocol import (
    CompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    UsageInfo,
    random_uuid,
)

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template

    _fastchat_available = True
except ImportError:
    _fastchat_available = False

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = logging.getLogger(__name__)
served_model = "meta-llama/Llama-2-7b-chat-hf"
app = fastapi.FastAPI()


MODEL_DIR = "/model"

image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install("torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("vllm")
    .pip_install("fschat")
    .pip_install("transformer-engine")
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder, secret=Secret.from_name("huggingface"))
)

stub = Stub("vllm-openai", image=image)


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    import os

    snapshot_download(
        served_model,
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean.
@stub.cls(gpu="A100", secret=Secret.from_name("huggingface"))
class Model:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        # We also add additional system prompting to the model to help it output json correctly.
        self.engine = LLM(MODEL_DIR)

    @method()
    async def generate(
        self,
        prompt: str,
        request_id: str,
        created_time: int,
        model_name: str,
        request: ChatCompletionRequest,
        raw_request: Request,
    ):
        from vllm.outputs import RequestOutput
        from vllm.sampling_params import SamplingParams

        try:
            sampling_params = SamplingParams(
                n=request.n,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                max_tokens=request.max_tokens,
                best_of=request.best_of,
                top_k=request.top_k,
                ignore_eos=request.ignore_eos,
                use_beam_search=request.use_beam_search,
            )
        except ValueError as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

        result_generator = self.engine.generate(prompt, sampling_params, request_id)

        async def abort_request() -> None:
            await self.engine.abort(request_id)

        def create_stream_response_json(
            index: int,
            text: str,
            finish_reason: Optional[str] = None,
        ) -> str:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=text),
                finish_reason=finish_reason,  # type: ignore
            )
            response = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=served_model,
                choices=[choice_data],
            )
            response_json = response.json(ensure_ascii=False)

            return response_json

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            # First chunk with role
            for i in range(request.n):
                choice_data = ChatCompletionResponseStreamChoice(
                    index=i,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None,
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id, choices=[choice_data], model=model_name
                )
                data = chunk.json(exclude_unset=True, ensure_ascii=False)
                yield f"data: {data}\n\n"

            previous_texts = [""] * request.n
            previous_num_tokens = [0] * request.n
            async for res in result_generator:
                res: RequestOutput
                for output in res.outputs:
                    i = output.index
                    delta_text = output.text[len(previous_texts[i]) :]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    response_json = create_stream_response_json(
                        index=i,
                        text=delta_text,
                    )
                    yield f"data: {response_json}\n\n"
                    if output.finish_reason is not None:
                        response_json = create_stream_response_json(
                            index=i,
                            text="",
                            finish_reason=output.finish_reason,
                        )
                        yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        # Streaming response
        if request.stream:
            background_tasks = BackgroundTasks()
            # Abort the request if the client disconnects.
            background_tasks.add_task(abort_request)
            return StreamingResponse(
                completion_stream_generator(),
                media_type="text/event-stream",
                background=background_tasks,
            )

        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await abort_request()
                return create_error_response(
                    HTTPStatus.BAD_REQUEST, "Client disconnected"
                )
            final_res = res
        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role="assistant", content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        if request.stream:
            # When user requests streaming but we don't stream, we still need to
            # return a streaming response with a single event.
            response_json = response.json(ensure_ascii=False)

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                fake_stream_generator(), media_type="text/event-stream"
            )

        return response


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


async def get_gen_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`"
        )

    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    return prompt


@app.get("/v1/model")
async def get_model():
    return {"model": served_model}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """

    logger.info(f"Received chat completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(
            HTTPStatus.BAD_REQUEST, "logit_bias is not currently supported"
        )

    prompt = await get_gen_prompt(request)

    model = Model()

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())

    return await model.generate(
        prompt=prompt,
        request_id=request_id,
        created_time=created_time,
        model_name=model_name,
        request=request,
        raw_request=raw_request,
    )


@stub.function(image=image, gpu="any")
@asgi_app()
def fastapi_app():
    return app
