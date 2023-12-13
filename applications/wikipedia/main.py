from itertools import product
import json
import subprocess
from pathlib import Path

from modal import Image, Stub, Volume, gpu, method, Secret

N_GPU = 50
N_INPUTS = 20
GPU_CONFIG = gpu.A10G()
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]

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

PUSH_TO_HUB = True
dataset_name = f"567-labs/wikipedia-embedding-{MODEL_SLUG}-sample"
dataset_file = "wiki-embeddings.paraquet"

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE * 512),
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


def generate_chunks_from_dataset(xs, chunk_size: int):
    for data in xs:
        id_ = data["id"]
        url = data["url"]
        title = data["title"]
        text = data["text"]
        for chunk_start in range(0, len(text), chunk_size):
            yield (
                id_,
                url,
                title,
                text[chunk_start : chunk_start + chunk_size],
            )


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
    async def embed(self, chunks):
        """Embeds a list of texts.  id, url, title, text = chunks[0]"""
        texts = [chunk[3] for chunk in chunks]
        res = await self.client.post("/embed", json={"inputs": texts})
        embeddings = res.json()
        return np.array(embeddings)


@stub.function(
    image=Image.debian_slim().pip_install("datasets", "pyarrow", "tqdm"),
    volumes={cache_dir: volume},
    timeout=5000,
    secret=Secret.from_name("huggingface-credentials"),
)
def embed_dataset(down_scale: float = 0.005, batch_size: int = 32):
    from datasets import load_from_disk, load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm
    import time
    import datetime
    import os

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
    batches = generate_batches(text_chunks, batch_size=batch_size)

    start = time.perf_counter()
    materialized_batches = list(batches)
    print(
        f"Materialized {len(materialized_batches)} batches in {time.perf_counter()-start:.2f} seconds"
    )
    acc_chunks = []
    embeddings = []
    char_count = 0
    for batch_chunks, batch_embeddings in zip(
        materialized_batches,
        model.embed.map(materialized_batches, order_outputs=True),
    ):
        acc_chunks.extend(batch_chunks)
        embeddings.extend(batch_embeddings)

        # Counting all characters in the dataset
        char_count += sum(map(len, [chunk[3] for chunk in batch_chunks]))

    end = time.perf_counter()

    if PUSH_TO_HUB:
        print(f"Pushing to hub {dataset_name}")
        table = pa.Table.from_arrays(
            [
                pa.array([chunk[0] for chunk in acc_chunks]),  # id
                pa.array([chunk[1] for chunk in acc_chunks]),  # url
                pa.array([chunk[2] for chunk in acc_chunks]),  # title
                pa.array([chunk[3] for chunk in acc_chunks]),  # text
                pa.array(embeddings),
            ],
            names=["id", "url", "title", "text", "embedding"],
        )
        pq.write_table(table, dataset_file)
        dataset = load_dataset("parquet", data_files=dataset_file)
        dataset.push_to_hub(dataset_name, token=os.environ["HUGGINGFACE_TOKEN"])

    duration = end - start
    characters_per_second = int(char_count / duration)
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
