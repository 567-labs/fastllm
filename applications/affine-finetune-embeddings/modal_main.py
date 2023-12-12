from main import run_optuna
import modal
import pathlib
from model import SimilarityModel

image = modal.Image.debian_slim().pip_install(
    "torch",
    "pytorch_lightning",
    "torchmetrics",
    "optuna",
    "pandas",
    "scikit-learn",
    "transformers",
    "tensorboard",
    "datasets",
)
stub = modal.Stub("run-optuna")
stub.volume = modal.Volume.new()

checkpoints_dirpath = pathlib.Path("/root/checkpoints")


@stub.function(image=image, gpu="any", volumes={checkpoints_dirpath: stub.volume})
def run():
    run_optuna(checkpoints_dirpath)
    stub.volume.commit()
    model = SimilarityModel.load_from_checkpoint(checkpoints_dirpath / "checkpoint-0.ckpt")
    res = model(["hello world"])
    print('embedding:', res)
    


@stub.local_entrypoint()
def main():
    run.remote()
