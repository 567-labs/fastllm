from main import run_optuna
import modal
import pathlib
from inference import inference

image = modal.Image.debian_slim().pip_install(
    "torch",
    "pytorch_lightning",
    "torchmetrics",
    "optuna",
    "pandas",
    "scikit-learn",
    "transformers",
    "tensorboard",
)
stub = modal.Stub("run-optuna")
stub.volume = modal.Volume.new()

checkpoints_dirpath = pathlib.Path("/root/checkpoints")


@stub.function(image=image, gpu="any", volumes={checkpoints_dirpath: stub.volume})
def run():
    run_optuna(str(checkpoints_dirpath))
    stub.volume.commit()
    res = inference(["hello world"], ["hi earth"], checkpoints_dirpath / "checkpoint-0.ckpt")
    print('embedding:', res)
    


@stub.local_entrypoint()
def main():
    run.remote()
