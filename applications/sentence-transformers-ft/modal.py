from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import modal
import pathlib
from finetune import finetune


# Modal constants
VOL_MOUNT_PATH = pathlib.Path("/vol")
GPU_CONFIG = "a10g"


# Functions for Modal Image build steps (cache the model and dataset)
# NOTE: this can be removed for simplicity, this is currently here for faster devloop times when repeatedly running the function
def download_model():
    model_id = "BAAI/bge-small-en-v1.5"
    SentenceTransformer(model_id)


def download_dataset():
    dataset_id = "quora"
    load_dataset(dataset_id, split="train")


# Modal resources
# TODO: uncomment this to make the volume persisted
# volume = modal.Volume.persisted(
#     f"sentence-transformers-ft-{int(datetime.now().timestamp())}"
# )
# non-persistent volume for dev purposes
volume = modal.Volume.new()
stub = modal.Stub("finetune-embeddings")
image = (
    modal.Image.debian_slim()
    .pip_install("sentence-transformers", "torch", "datasets")
    .run_function(download_model)
    .run_function(download_dataset)
)


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=15000,
    volumes={VOL_MOUNT_PATH: volume},
    _allow_background_volume_commits=True,
)
def finetune_modal():
    model_id = "BAAI/bge-small-en-v1.5"
    dataset_fraction = 300

    finetune(
        model_id=model_id, dataset_fraction=dataset_fraction, save_path=VOL_MOUNT_PATH
    )


# run on modal with `modal run main.py`
@stub.local_entrypoint()
def main():
    finetune_modal.remote()
