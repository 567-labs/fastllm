# source: https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/vllm_inference.py
from modal import Stub, Image, Secret, method, asgi_app
from typing import List
import os
import json

import fastapi
from pydantic import BaseModel

app = fastapi.FastAPI(
    title="vLLM",
)


def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "meta-llama/Llama-2-13b-chat-hf",
        local_dir="/model",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


MODEL_DIR = "/model"

image = (
    Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install("torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118")
    # Pin vLLM to 07/19/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@bda41c70ddb124134935a90a0d51304d2ac035e8"
    )
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder, secret=Secret.from_name("huggingface"))
)

stub = Stub("vllm", image=image)


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
        self.llm = LLM(MODEL_DIR)
        self.template = """SYSTEM: Always correctly output response data as correctly formatted json in a codeblock\n{system}
USER: {input}
ASSISTANT: ```json\n"""

    @method()
    def generate(
        self,
        system: str,
        inputs: List[str],
        max_tokens: int = 800,
        temperature: float = 0.1,
        presence_penalty: float = 1.15,
    ):
        from vllm import SamplingParams

        prompts = [self.template.format(system=system, input=ii) for ii in inputs]
        sampling_params = SamplingParams(
            temperature=temperature,
            # we add a ``` to the end of the prompt to ensure the model outputs a codeblock
            # improving the chances of it outputting correctly formatted json
            stop="```",
            top_p=1,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
        )
        result = self.llm.generate(prompts, sampling_params)
        num_tokens = 0
        results = [output.outputs[0].text for output in result]
        num_tokens = sum([len(output.outputs[0].token_ids) for output in result])
        return results, num_tokens


class InputModel(BaseModel):
    system: str
    data: List[str]
    max_tokens: int = 800
    temperature: float = 0.1
    presence_penalty: float = 1.15


@app.post("/")
def main(input: InputModel):
    def try_json(x):
        try:
            return json.loads(x)
        except Exception as e:
            print(e)
            return x

    model = Model()
    data, num_tokens = model.generate.call(
        system=input.system,
        inputs=input.data,
        max_tokens=input.max_tokens,
        temperature=input.temperature,
        presence_penalty=input.presence_penalty,
    )
    return {
        "data": [try_json(x) for x in data],
        "num_tokens": num_tokens,
    }


@stub.function(image=image)
@asgi_app()
def fastapi_app():
    return app
