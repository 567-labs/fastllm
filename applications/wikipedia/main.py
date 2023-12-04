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


def generate_chunks_from_dataset(xs, chunk_size=400):
    for data in xs:
        text = data["text"]
        for chunk_start in range(0, len(text), chunk_size):
            yield text[chunk_start : chunk_start + chunk_size]


def generate_batches(xs, batch_size=50):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@stub.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    # Use up to 10 GPU containers at once.
    concurrency_limit=10,
    # Allow each container to process up to 10 batches at once.
    allow_concurrent_inputs=40,
)
class TextEmbeddingsInference:
    def __enter__(self):
        # If the process is running for a long time, the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000")

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.process.terminate()

    @method()
    async def embed(self, texts: list[str]):
        n_chars = sum(map(len, texts))
        _ = await self.client.post("/embed", json={"inputs": texts})
        return n_chars


@stub.function(
    image=Image.debian_slim().pip_install("datasets"),
    volumes={cache_dir: volume},
    timeout=5000,
)
def embed_dataset():
    from datasets import load_from_disk

    import time

    start = time.time()
    # Load the dataset as a Hugging Face dataset
    print("Loading dataset from disk... ~ 40 seconds")
    dataset = load_from_disk(f"{cache_dir}/wikipedia")
    print(f"Dataset loaded in {time.time()-start:.2f} seconds")

    # Extract the total size of the dataset
    ttl_size = len(dataset["train"])
    sample_size = int(ttl_size * 0.01)
    print(f"Calculated dataset size of {ttl_size} and sample size of {sample_size}")

    # Iterate over the first 5% of the dataset's rows
    subset = dataset["train"].select(range(sample_size))
    model = TextEmbeddingsInference()

    print(f"Working with {sample_size} rows")

    text_chunks = generate_chunks_from_dataset(subset, chunk_size=400)
    batches = generate_batches(
        text_chunks, batch_size=32
    )  # 32 is the max batch size of the model

    start = time.perf_counter()
    counter = 0
    for n_chars in model.embed.map(batches, order_outputs=False):
        counter += n_chars
    end = time.perf_counter()
    print(f"Processed {counter} characters in {end-start:.2f} seconds")
    print(f"Throughput: {counter/(end-start):.2f} characters per second")


@stub.local_entrypoint()
def main():
    embed_dataset.remote()
