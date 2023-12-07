from main import run_optuna
import modal
import pathlib

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

p = pathlib.Path("/root/checkpoints")


@stub.function(image=image, gpu="any")
def run():
    run_optuna()


@stub.local_entrypoint()
def main():
    run.remote()
