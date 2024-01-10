from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import modal
import pathlib
from finetune_OnlineContrastiveLoss import finetune
from datetime import datetime

MODEL_ID = "BAAI/bge-base-en-v1.5"

# Modal constants
VOL_MOUNT_PATH = pathlib.Path("/vol")
GPU_CONFIG = "a100"
USE_CACHED_IMAGE = True  # enable this to download the dataset and base model into the image for faster repeated runs
PERSIST_VOLUME = True  # Enable this to persist the trained model in Modal afterwards


def download_model():
    SentenceTransformer(MODEL_ID)


def download_dataset():
    dataset_id = "quora"
    load_dataset(dataset_id, split="train")


# Modal resources
if PERSIST_VOLUME:
    # Persisted volumes are available after the modal application finishes
    volume = modal.Volume.persisted(
        f"sentence-transformers-ft-{int(datetime.now().timestamp())}"
    )
else:
    volume = modal.Volume.new()
stub = modal.Stub("finetune-embeddings")
image = modal.Image.debian_slim().pip_install(
    "sentence-transformers", "torch", "datasets"
)
if USE_CACHED_IMAGE:
    image = image.run_function(download_model).run_function(download_dataset)


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=30000,
    volumes={VOL_MOUNT_PATH: volume},
    _allow_background_volume_commits=True,
)
def finetune_modal():
    score, model = finetune(
        model_id=MODEL_ID, dataset_fraction=5, epochs=8, save_path=VOL_MOUNT_PATH
    )

    # Move model to CPU so it can be loaded to the host computer (currently is on CUDA)
    model.to("cpu")

    return score, model


# run on modal with `modal run main-modal.py`
@stub.local_entrypoint()
def main():
    score, model = finetune_modal.remote()

    print("Post Train eval score", score)
