from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import modal
import pathlib
from finetune_OnlineContrastiveLoss import finetune

MODEL_ID = "BAAI/bge-small-en-v1.5"

# Modal constants
VOL_MOUNT_PATH = pathlib.Path("/vol")
GPU_CONFIG = "a10g"
USE_CACHED_IMAGE = True  # enable this to download the dataset and base model into the image for faster repeated runs


def download_model():
    SentenceTransformer(MODEL_ID)


def download_dataset():
    dataset_id = "quora"
    load_dataset(dataset_id, split="train")


# Modal resources
# TODO: uncomment this to make the volume persisted
# TODO: for final code just make it persisted, right now leave it commented for dev purposes
# volume = modal.Volume.persisted(
#     f"sentence-transformers-ft-{int(datetime.now().timestamp())}"
# )
# non-persistent volume for dev purposes
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
    timeout=15000,
    volumes={VOL_MOUNT_PATH: volume},
    _allow_background_volume_commits=True,
)
def finetune_modal():
    finetune(model_id=MODEL_ID, dataset_fraction=2, save_path=VOL_MOUNT_PATH)


# run on modal with `modal run main.py`
@stub.local_entrypoint()
def main():
    finetune_modal.remote()
