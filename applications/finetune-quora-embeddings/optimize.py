from modal import Stub, Image, gpu, Volume, NetworkFileSystem
from optuna import Trial
from helpers.data import format_dataset, score_prediction
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
import pandas as pd
import math
from typing import List

# GPU Configuration
gpu_config = gpu.A100()

# Finetuning Configuration ( Arrays are configurable parameters )
MODELS = [
    # "BAAI/bge-base-en-v1.5",
    "WhereIsAI/UAE-Large-V1"
]
DENSE_LAYER_DIMS = [2048]
ACTIVATION_FUNCTIONS = ["Tanh"]  # This is a torch func, we use getattr to read it
SCHEDULER = ["warmupconstant"]
DATASET_SIZE = [128000]
WARMUP_STEPS = [500]
BATCH_SIZE = 64
MODEL_SAVE_PATH = "/output"
OPTIMIZATION_METRIC_FUNC = roc_auc_score

STUDY_NFS = NetworkFileSystem.persisted("modal-optimization")
JOURNAL_PATH = "/root/cache/journal.log"
STUDY_NAME = "optuna-optimization"

WORKERS = 1
NUM_TRIALS = 1

# DATASET CONFIG
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.persisted("datasets")
CACHE_DIRECTORY = f"{DATASET_DIR}/cached-embeddings"

# Eval Configuration
METRICS = {
    "accuracy": accuracy_score,  # This is the number of correct predictions by the model ( TP + TN )/ (# of samples)
    "precision": precision_score,  # This measures the number of positive class predicitons (TP) / (TP + FP)
    "recall": recall_score,  # This measure the number of negative class predictions (TP) / ( TP + FN )
    "AUC": roc_auc_score,
}

# Final Result
RESULT_FILE = "optimize.json"
DATAFRAME_PICKLE_PATH = "optimize_df.pkl"

stub = Stub("modal-optimization")


def download_model():
    from sentence_transformers import SentenceTransformer

    for model in MODELS:
        SentenceTransformer(model)


image = (
    Image.debian_slim()
    .pip_install("sentence-transformers", "torch", "datasets", "optuna", "pandas")
    .run_function(download_model)
)


def objective(trial: Trial, existing_experiments: List[dict]):
    from sentence_transformers import SentenceTransformer, losses, evaluation, models
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    import shutil
    import os

    # delete the previously saved model if it exists
    if os.path.exists(MODEL_SAVE_PATH):
        # Delete the directory if it exists
        shutil.rmtree(MODEL_SAVE_PATH)

    params = {
        "model": trial.suggest_categorical("model", MODELS),
        "dense_out_features": trial.suggest_categorical(
            "dense_out_features", DENSE_LAYER_DIMS
        ),
        "activation_function": trial.suggest_categorical(
            "activation_function", ACTIVATION_FUNCTIONS
        ),
        "scheduler": trial.suggest_categorical("scheduler", SCHEDULER),
        "dataset_size": trial.suggest_categorical("dataset_size", DATASET_SIZE),
        "epochs": 8,
        "warmup_steps": trial.suggest_categorical("warmup_steps", WARMUP_STEPS),
    }
    # We load our model and freeze the layers ( see https://github.com/UKPLab/sentence-transformers/issues/680 )
    embedding_model = SentenceTransformer(params["model"])
    auto_model = embedding_model._first_module().auto_model
    for param in auto_model.parameters():
        param.requires_grad = False

    dense_model = models.Dense(
        in_features=embedding_model.get_sentence_embedding_dimension(),
        out_features=params["dense_out_features"],
        activation_function=getattr(nn, params["activation_function"])(),
    )

    model = SentenceTransformer(modules=[embedding_model, dense_model])
    print(
        f"Initialized Sentence Transformer model and running job with params of {params}"
    )
    # Load in the dataset
    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    train_dataset = dataset["train"].select(range(params["dataset_size"]))
    test_dataset = dataset["test"]

    train_loss = losses.OnlineContrastiveLoss(model)
    train_examples = format_dataset(train_dataset)
    test_examples = format_dataset(test_dataset)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples, batch_size=BATCH_SIZE
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=params["epochs"],
        warmup_steps=params["warmup_steps"],
        save_best_model=True,
        show_progress_bar=True,
        output_path=MODEL_SAVE_PATH,
        scheduler=params["scheduler"],
    )

    model = SentenceTransformer(MODEL_SAVE_PATH)

    predictions, test_labels = score_prediction(model, train_dataset, test_dataset)

    eval_results = {
        metric: function(test_labels, predictions)
        for metric, function in METRICS.items()
    }

    existing_experiments.append(eval_results | params)
    print("Logging results so far")
    for idx, experiment in enumerate(existing_experiments):
        print(f"Idx: {idx}, Result: {experiment}")

    return OPTIMIZATION_METRIC_FUNC(test_labels, predictions)


@stub.function(
    image=image,
    gpu=gpu_config,
    network_file_systems={"/root/cache": STUDY_NFS},
    volumes={DATASET_DIR: DATASET_VOLUME},
    timeout=86400,
)
def optimize_hyperparameters(n_trials: int):
    import optuna
    import os

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(JOURNAL_PATH)
    )
    results = []
    # Set it so that we can resume a study in the event that something fails
    study = optuna.create_study(
        study_name="finetuning", storage=storage, load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, results), n_trials=n_trials)

    return results


@stub.function(
    image=image,
    network_file_systems={"/root/cache": STUDY_NFS},
)
def get_dataframe():
    import optuna

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(JOURNAL_PATH)
    )

    # Set it so that we can resume a study in the event that something fails
    study = optuna.create_study(
        study_name="finetuning", storage=storage, load_if_exists=True
    )

    return study.trials_dataframe()


@stub.function(
    image=image,
    network_file_systems={"/root/cache": STUDY_NFS},
)
def reset_study():
    import os

    if os.path.exists(JOURNAL_PATH):
        print(f"Deleting {JOURNAL_PATH}")
        os.remove(JOURNAL_PATH)


@stub.local_entrypoint()
def main():
    import os

    reset_study.remote()
    results = []
    trials_per_worker = math.ceil(NUM_TRIALS / WORKERS)

    for resp in optimize_hyperparameters.map(
        [trials_per_worker for _ in range(WORKERS)],
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(resp, Exception):
            print(f"Encountered Exception of {resp}")
            continue

        results.extend(resp)

    with open(RESULT_FILE, "a+") as json_file:
        for result in results:
            json_file.write(json.dumps(result, indent=2) + "\n")

    df = get_dataframe.remote()
    if os.path.exists(DATAFRAME_PICKLE_PATH):
        prev_df = pd.read_pickle(DATAFRAME_PICKLE_PATH)
        df = pd.concat([prev_df, df])
    df.to_pickle(DATAFRAME_PICKLE_PATH)
