from main import run_optuna
import modal
import pathlib
from model import SimilarityModel
from main import objective
import optuna
from optuna.study import StudyDirection

N_GPU = 3

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
nfs_volume = modal.NetworkFileSystem.new()
nfs_path = "/root/cache"
study_name = "parallel optuna"

# concurrent multi gpu
@stub.function(
    image=image,
    volumes={checkpoints_dirpath: stub.volume},
    network_file_systems={"/root/cache": nfs_volume},
)
def initialize():
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(nfs_path + "/journal.log")
    )
    storage.create_new_study(study_name=study_name,directions=[StudyDirection.MAXIMIZE])

@stub.function(
    image=image,
    gpu="any",
    volumes={checkpoints_dirpath: stub.volume},
    network_file_systems={"/root/cache": nfs_volume},
    concurrency_limit=3,
)
def run(i:int):
    print('starting gpu container:', i)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(nfs_path + "/journal.log"),
    )
    study = optuna.load_study(
        study_name=study_name,
        storage=storage
    )
    study.optimize(lambda trial: objective(trial, checkpoints_dirpath), n_trials=5)
    stub.volume.commit()
    trials = study.get_trials()

    return trials, i


@stub.local_entrypoint()
def main():
    initialize.remote()
    # Run Optuna optimization
    for r in run.map(range(1,4)):
        res, i = r
        print(f"Finished training on gpu container {i}.\nTrials: {res}")

# ------------------------------------------------ #
# # single gpu
# @stub.function(
#     image=image,
#     gpu="any",
#     volumes={checkpoints_dirpath: stub.volume},
# )
# def run():
#     run_optuna(checkpoints_dirpath)
#     stub.volume.commit()
#     model = SimilarityModel.load_from_checkpoint(
#         checkpoints_dirpath / "checkpoint-0.ckpt"
#     )
#     res = model(["hello world"])
#     print("embedding:", res)

# @stub.local_entrypoint()
# def main():
#     run.remote()
