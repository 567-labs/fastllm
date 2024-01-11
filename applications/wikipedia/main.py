import asyncio
import json
import subprocess
from modal import Image, Secret, Stub, Volume, gpu, method

# We first set out configuration variables for our script.
## Embedding Containers Configuration
N_GPU = 100
GPU_CONFIG = gpu.A10G()
MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]
BATCH_SIZE = 512
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"  # Ampere 86 for A10s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.4.0" # Ampere 80 for A100s.
    # "ghcr.io/huggingface/text-embeddings-inference:0.3.0"  # Turing for T4s.
)

## Dataset-Specific Configuration
DATASET_NAME = "wikipedia"
DATASET_READ_VOLUME = Volume.persisted("embedding-wikipedia")
EMBEDDING_CHECKPOINT_VOLUME = Volume.persisted("checkpoint")
DATASET_DIR = "/data"
CHECKPOINT_DIR = "/checkpoint"
SAVE_TO_DISK = True

## Upload-Specific Configuration
DATASET_HF_UPLOAD_REPO_NAME = "567-labs/upload-test"
UPLOAD_TO_HF = True

## HF Text-Embedding Inference specific Configuration

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


stub = Stub("embeddings")


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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    spawn_server()


tei_image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("httpx", "transformers")
    .run_function(download_model, gpu=GPU_CONFIG)
)

with tei_image.imports():
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


def generate_batches(xs, batch_size):
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
    retries=3,
)
class TextEmbeddingsInference:
    def __enter__(self):
        # If the process is running for a long time, the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.process.terminate()

    async def _embed(self, chunk_batch, retries=1):
        texts = [chunk[3] for chunk in chunk_batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return np.array(res.json())

    def _filter_large_tokens(self, chunks):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        new_chunk_batch = []
        for chunk in chunks:
            # Split text into two portions since max input size of encoder is 512 tokens
            s1 = chunk[3][: len(chunk[3]) // 2]
            s2 = chunk[3][len(chunk[3]) // 2 :]
            tokens = tokenizer.encode(s1) + tokenizer.encode(s2)

            if len(tokens) <= 512:
                new_chunk_batch.append(chunk)
            else:
                print(f"Identified issue with {chunk[3]}")
                new_chunk_batch.append(
                    [
                        chunk[0],
                        chunk[1],
                        chunk[2],
                        s1 if len(s1) <= 512 else s1[: len(s1) // 2],
                    ]
                )
        return new_chunk_batch

    @method()
    async def embed(self, chunks):
        """Embeds a list of texts.  id, url, title, text = chunks[0]"""
        coros = [
            self._embed(chunk_batch)
            for chunk_batch in generate_batches(
                self._filter_large_tokens(chunks), batch_size=BATCH_SIZE
            )
        ]

        coros_results = await asyncio.gather(*coros)
        # Filter out items that don't have a length of 384 and count them
        filtered_results = []
        invalid_items = []
        for batch in coros_results:
            if batch.ndim == 0:
                print(f"Found an empty batch of {batch}")
                continue
            filtered_batch = [item for item in batch if len(item) == 384]
            invalid_batch = [item for item in batch if len(item) != 384]
            filtered_results.extend(filtered_batch)
            invalid_items.extend(invalid_batch)
        embeddings = np.array(filtered_results)

        if len(invalid_items) > 0:
            print(f"Filtered out {len(invalid_items)} items")
            print(invalid_items)


        return chunks, embeddings


def load_dataset_from_disk(down_scale=0.01):
    import time
    from datasets import load_from_disk

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    dataset = load_from_disk(f"{DATASET_DIR}/wikipedia")
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds")

    # Extract the total size of the dataset
    ttl_size = len(dataset["train"])

    sample_size = int(ttl_size * down_scale)

    return dataset["train"].select(range(sample_size))


def save_dataset_to_intermediate_checkpoint(
    acc_chunks, embeddings, batch_size, down_scale
):
    import pyarrow as pa
    from datasets import Dataset
    import os
    import shutil

    assert (
        len(acc_chunks) == len(embeddings)
    ), f"Mismatch between chunks and generated embeddings - chunks: {len(acc_chunks)} items, embedding: {len(embeddings)} "
    print("Starting to generate data now")

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
    print(f"Created table of size {len(embeddings)}")
    path_parent_folder = f"{CHECKPOINT_DIR}/{MODEL_SLUG}-{batch_size}-{down_scale}"

    if os.path.exists(path_parent_folder):
        shutil.rmtree(path_parent_folder)
    dataset = Dataset(table)
    dataset.save_to_disk(path_parent_folder)
    EMBEDDING_CHECKPOINT_VOLUME.commit()
    print(f"Saved checkpoint at {path_parent_folder}")


def upload_result_to_hf(batch_size, down_scale):
    import os
    from huggingface_hub import HfApi
    import time

    path_parent_folder = f"{CHECKPOINT_DIR}/{MODEL_SLUG}-{batch_size}-{down_scale}"
    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        private=False,
        repo_type="dataset",
        exist_ok=True,
    )

    print(f"Pushing to hub {DATASET_HF_UPLOAD_REPO_NAME}")
    start = time.perf_counter()
    api.upload_folder(
        folder_path=path_parent_folder,
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        repo_type="dataset",
        multi_commits=True,
        multi_commits_verbose=True,
    )

    end = time.perf_counter()
    print(f"Uploaded in {end-start}s")


@stub.function(
    image=Image.debian_slim().pip_install(
        "datasets", "pyarrow", "hf_transfer", "huggingface_hub"
    ),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        CHECKPOINT_DIR: EMBEDDING_CHECKPOINT_VOLUME,
    },
    timeout=86400,
    secret=Secret.from_name("huggingface-credentials"),
)
def embed_dataset(down_scale: float = 0.005, batch_size: int = 512 * 50):
    import datetime
    import time

    if UPLOAD_TO_HF and not SAVE_TO_DISK:
        raise ValueError(
            "Uploading to HF requires SAVE_TO_DISK to be set to true in case of intermediate failure."
        )

    dataset_chars = 19560538957  # sum(map(len, dataset["train"]["text"]))
    subset = load_dataset_from_disk(down_scale)
    model = TextEmbeddingsInference()
    text_chunks = generate_chunks_from_dataset(subset, chunk_size=512)
    batches = generate_batches(text_chunks, batch_size=batch_size)

    start = time.perf_counter()
    acc_chunks = []
    embeddings = []
    for resp in model.embed.map(batches, order_outputs=False, return_exceptions=True):
        if isinstance(resp, Exception):
            print(f"Exception: {resp}")
            continue

        batch_chunks, batch_embeddings = resp

        acc_chunks.extend(batch_chunks)
        embeddings.extend(batch_embeddings)

    print(f"Generated a total of {len(embeddings)}")
    end = time.perf_counter()

    duration = end - start
    characters = sum(map(len, [chunk[3] for chunk in acc_chunks]))
    characters_per_sec = int(characters / duration)
    extrapolated_duration_cps_fmt = str(
        datetime.timedelta(seconds=dataset_chars / characters_per_sec)
    )
    resp = {
        "downscale": down_scale,
        "batch_size": batch_size,
        "n_gpu": N_GPU,
        "duration_mins": duration / 60,
        "characters_per_sec": characters_per_sec,
        "extrapolated_duration": extrapolated_duration_cps_fmt,
    }
    print(f"Embedding Job Completed. Here are the stats: \n{json.dumps(resp)}")

    if SAVE_TO_DISK:
        print("Starting to save intermediate checkpoint to disk")
        save_dataset_to_intermediate_checkpoint(
            acc_chunks, embeddings, batch_size, down_scale
        )

    if UPLOAD_TO_HF:
        upload_result_to_hf(batch_size, down_scale)

    return resp


@stub.local_entrypoint()
def main():
    scale = 1
    batch_size = 512 * 150
    with open("benchmarks.json", "a") as f:
        benchmark = embed_dataset.remote(down_scale=scale, batch_size=batch_size)
        f.write(json.dumps(benchmark, indent=2) + "\n")
