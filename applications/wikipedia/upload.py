from pathlib import Path
import time
from modal import Image, Stub, Volume, Secret

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_SLUG = MODEL_ID.split("/")[-1]

BATCH_SIZE = 512
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

dataset_name = "567-labs/upload-test-benchmark"
dataset_file = "wiki-embeddings.parquet"


stub = Stub("embeddings")


@stub.function(
    image=Image.debian_slim().pip_install(
        "datasets", "pyarrow", "tqdm", "hf_transfer", "huggingface_hub"
    ),
    volumes={cache_dir: volume},
    _allow_background_volume_commits=True,
    timeout=84600,
    secret=Secret.from_name("huggingface-credentials"),
)
def upload_dataset():
    from huggingface_hub import HfApi
    import os
    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(repo_id=dataset_name, private=False, repo_type="dataset",exist_ok=True)


    print(f"Pushing to hub {dataset_name}")
    start = time.perf_counter()
    api.upload_folder(
        folder_path=f"{cache_dir}/wikipedia",
        repo_id = dataset_name,
        repo_type="dataset",
        allow_patterns="*.arrow",
        multi_commits=True,
        multi_commits_verbose=True
    )
    

    end = time.perf_counter()
    print(f"Uploaded in {end-start}s")


@stub.local_entrypoint()
def main():
    upload_dataset.remote()
