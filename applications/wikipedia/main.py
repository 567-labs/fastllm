from itertools import product
import json
import subprocess
from pathlib import Path

from modal import Image, Stub, Volume, gpu, method

N_GPU = 50
N_INPUTS = 30
GPU_CONFIG = gpu.A10G()
MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 256 * 2
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
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(16384 * 3),
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
    concurrency_limit=N_GPU,
    # Allow each container to process up to 10 batches at once.
    allow_concurrent_inputs=N_INPUTS,
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
def embed_dataset(down_scale: float = 0.005, batch_size: int = 32):
    from datasets import load_from_disk

    import time
    import datetime

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print("Loading dataset from disk... ~ 40 seconds")
    dataset = load_from_disk(f"{cache_dir}/wikipedia")
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds")

    # Extract the total size of the dataset
    ttl_size = len(dataset["train"])

    # Counting all characters in the dataset
    dataset_chars = 19560538957  # sum(map(len, dataset["train"]["text"]))
    print(f"Total dataset characters: {dataset_chars}")

    sample_size = int(ttl_size * down_scale)
    print(f"Calculated dataset size of {ttl_size} and sample size of {sample_size}")

    # Iterate over the first 5% of the dataset's rows
    subset = dataset["train"].select(range(sample_size))
    model = TextEmbeddingsInference()

    print(f"Working with {sample_size} rows")

    text_chunks = generate_chunks_from_dataset(subset, chunk_size=400)
    batches = generate_batches(
        text_chunks, batch_size=batch_size
    )  # 32 is the max batch size of the model

    start = time.perf_counter()
    materialized_batchs = list(batches)
    print(
        f"Materialized {len(materialized_batchs)} batches in {time.perf_counter()-start:.2f} seconds"
    )

    start = time.perf_counter()
    counter = 0
    for n_chars in model.embed.map(materialized_batchs, order_outputs=False):
        counter += n_chars
    end = time.perf_counter()

    duration = end - start
    characters_per_second = int(counter / duration)
    extrapolated_duration = int(duration / down_scale)
    extrapolated_duration_fmt = str(datetime.timedelta(seconds=extrapolated_duration))
    extrapolated_duration_tps_fmt = str(
        datetime.timedelta(seconds=dataset_chars / characters_per_second)
    )
    return {
        "downscale": down_scale,
        "batch_size": batch_size,
        "n_gpu": N_GPU,
        "n_inputs": N_INPUTS,
        "duration": duration,
        "characters_per_second": characters_per_second,
        "extrapolated_duration": extrapolated_duration,
        "extrapolated_duration_fmt": extrapolated_duration_fmt,
        "extrapolated_duration_tps_fmt": extrapolated_duration_tps_fmt,
    }


@stub.local_entrypoint()
def main():
    for scale, batch_size in product([0.001], [256, 512]):
        with open(f"benchmarks.json", "a") as f:
            benchmark = embed_dataset.remote(down_scale=scale, batch_size=batch_size)
            print(json.dumps(benchmark, indent=2))
            f.write(json.dumps(benchmark, indent=2) + "\n")
