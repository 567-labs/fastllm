""" 
This script is to test downloading a model from the HuggingFace Hub 
and persisting it to a volume.
"""
from modal import Image, Stub, Volume, Secret

MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32


def download_model():
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id="BAAI", filename="bge-base-en-v1.5")


stub = Stub("embeddings")

vol = Volume.persisted("embeddings")


def download_data():
    from datasets import load_dataset, load_from_disk

    # "20210301.en" use a smaller dataset for testing
    WIKI, SET = (
        "wikipedia",
        "20220301.ab",
    )
    PATH = f"/data/{WIKI}-{SET}"

    # check if dataset is already downloaded
    try:
        dataset = load_from_disk(PATH)
        return dataset
    except FileNotFoundError:
        print("Dataset not found, downloading...")
        pass

    dataset = load_dataset(WIKI, SET)
    dataset.save_to_disk(PATH)
    vol.commit()
    return dataset


@stub.function(
    image=Image.debian_slim().pip_install("tqdm", "datasets", "apache-beam"),
    volumes={"/data": vol},
    timeout=300 * 6,
)
def embed_dataset():
    # >>> modal run download.py::embed_dataset
    dataset = download_data()
    print(f"{len(dataset)=}")
