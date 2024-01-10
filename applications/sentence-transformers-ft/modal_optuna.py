from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import modal
import pathlib
import optuna
from finetune_OnlineContrastiveLoss import finetune
from datetime import datetime
import torch


# Modal constants
GPU_CONFIG = "a10g"
USE_CACHED_IMAGE = True  # enable this to download the dataset and base model into the image for faster repeated runs

#### Total number of trials = N_GPU * N_TRIALS
N_GPU = 10
N_TRIALS = 5  # Number of trials PER gpu


# Functions for Modal Image build steps (cache the dataset)
def download_dataset():
    dataset_id = "quora"
    load_dataset(dataset_id, split="train")


# Modal resources
stub = modal.Stub("finetune-embeddings-optuna")
image = modal.Image.debian_slim().pip_install(
    "sentence-transformers", "torch", "datasets", "optuna", "pandas"
)
if USE_CACHED_IMAGE:
    image = image.run_function(download_dataset)


#### We use a modal NFS resource to for the backend storage system for Optuna
stub.nfs_volume = modal.NetworkFileSystem.new()
JOURNAL_PATH = "/root/cache/journal.log"
STUDY_NAME = "sentence-transformers-ft study"

#### We store the actual trial results in a separate persistent Modal volume
VOL_MOUNT_PATH = pathlib.Path("/vol")
volume = modal.Volume.persisted(
    f"sentence-transformers-ft-optuna-{int(datetime.now().timestamp())}"
)


@stub.function(image=image, network_file_systems={"/root/cache": stub.nfs_volume})
def initialize_optuna():
    #### We initialize the JournalStorage object with NFS which is used as the backend for Optuna
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(JOURNAL_PATH)
    )
    #### Create the study object inside JournalStorage which is used to track Optuna trials
    storage.create_new_study(
        study_name=STUDY_NAME, directions=[optuna.study.StudyDirection.MAXIMIZE]
    )


def objective(trial: optuna.Trial):
    #### The objective function for optuna, simply calls the finetune function with various hyperparameters

    # Optuna Hyperparameters
    dense_out_features = trial.suggest_int("dense_out_features", 100, 700)
    activation_function_str = trial.suggest_categorical("activation", ["Tanh", "ReLU"])
    activation_function = getattr(torch.nn, activation_function_str)()
    epochs = trial.suggest_int("epochs", 7, 12)
    scheduler = trial.suggest_categorical(
        "scheduler", ["warmupconstant", "warmuplinear", "warmupcosine"]
    )
    model_id = trial.suggest_categorical(
        "model_id",
        [
            "BAAI/bge-small-en-v1.5",
            # "BAAI/bge-base-en-v1.5",
            # "BAAI/bge-large-en-v1.5",
            "thenlper/gte-small",
            "intfloat/e5-small-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
        ],
    )

    # TODO: add dropout, seems kinda annoying tho https://github.com/UKPLab/sentence-transformers/issues/677

    eval_score, _ = finetune(
        model_id=model_id,
        # all finetuning trials results will be saved in modal volume
        save_path=VOL_MOUNT_PATH / f"trial-{trial.number}",
        dense_out_features=dense_out_features,
        epochs=epochs,
        dataset_fraction=5,
        activation_function=activation_function,
        scheduler=scheduler,
    )
    # return the eval_score as Optuna is optimizing for this
    return eval_score


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=30000, # if timeout is too low, study may not finish
    volumes={VOL_MOUNT_PATH: volume},
    _allow_background_volume_commits=True,
    network_file_systems={"/root/cache": stub.nfs_volume},
    concurrency_limit=N_GPU,
)
def run_optuna(i: int):
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(JOURNAL_PATH)
    )
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)

    study.optimize(lambda trial: objective(trial), n_trials=N_TRIALS)

    trials = study.get_trials()

    print("------trials------\n", trials)
    return i


@stub.function(
    image=image,
    network_file_systems={"/root/cache": stub.nfs_volume},
    volumes={VOL_MOUNT_PATH: volume},
    _allow_background_volume_commits=True,
)
def conclude_optuna():
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(JOURNAL_PATH)
    )
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)

    # Log trials to Modal
    trials = study.get_trials()
    print("### All Trials ###\n", trials)

    # Saves the trials as a CSV object in Modal volume.
    # Can be retrieved using: `modal volume get sentence-transformers-ft-optuna-<timestamp> /trials.csv .`
    df = study.trials_dataframe()
    df.to_csv(VOL_MOUNT_PATH / "trials.csv")

    best_trial = study.best_trial
    print("### Best Trial ID:", best_trial._trial_id)
    return trials


@stub.local_entrypoint()
def main():
    initialize_optuna.remote()
    # Run distributed Optuna optimization using Modal's map() function
    for i in run_optuna.map(range(1, N_GPU + 1)):
        print(f"Finished training on gpu container {i}.")
    print(conclude_optuna.remote())
