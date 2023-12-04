import subprocess
from pathlib import Path
from modal import Image, Stub, Volume, gpu, method

GPU_CONFIG = gpu.A10G()
MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4s.
)
dataset_name = "wikipedia"
volume = Volume.persisted("embedding-wikipedia")
cache_dir = "/data"
data_dir = f"{cache_dir}/{dataset_name}"
DATA_PATH = Path(data_dir)

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
]


def spawn_server() -> subprocess.Popen:
    import socket

    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


def download_model():
    # Wait for server to start. This downloads the model weights when not present.
    spawn_server()


stub = Stub("embeddings")


tei_image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, gpu=GPU_CONFIG)
    .pip_install("httpx")
)


with tei_image.run_inside():
    import numpy as np


@stub.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    # Use up to 20 GPU containers at once.
    concurrency_limit=10,
    # Allow each container to process up to 10 batches at once.
    allow_concurrent_inputs=10,
)
class TextEmbeddingsInference:
    def __enter__(self):
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000")

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.process.terminate()

    @method()
    async def embed(self, inputs: list[str]):
        print(f"Recieved {len(inputs)} inputs")
        resp = self.client.post("/embed", json={"inputs": inputs})
        resp = await resp
        resp.raise_for_status()
        outputs = resp.json()

        # Returning a list is slower because of additional Modal-specific overhead,
        # to be fixed shortly.
        return np.array(outputs)


@stub.function(
    image=Image.debian_slim().pip_install("datasets"),
    volumes={cache_dir: volume},
    timeout=5000,
)
def embed_dataset():
    from datasets import load_from_disk

    print("Starting Model Embedding")
    import time

    start = time.perf_counter()
    dataset = load_from_disk(f"{cache_dir}/wikipedia")
    print(f"Loaded dataset in {time.perf_counter()-start:.2f}s")  # 40s
    subset = dataset["train"].select(range(5))

    model = TextEmbeddingsInference()

    # Text Embedding can only take in 512 tokens at a given time ( Input validation error: `inputs` must have less than 512 tokens. Given: 598 )
    # Therefore -> Move towards migrating the
    count = 0

    def generate_batches():
        nonlocal count
        for item in subset:
            sentences = item["text"].split(". ")
            for sentence in sentences:
                count += 1
                if len(sentence) > 400:
                    continue
                yield [sentence]

    start = time.time()
    for output_batch in model.embed.map(generate_batches(), order_outputs=False):
        # Do something with the outputs.
        print(f"Generated {len(output_batch)} embeddings")
    end = time.time()
    print(f"Took {end-start}s to embed {count} sentences")


@stub.local_entrypoint()
def main():
    embed_dataset.remote()
