from modal import Stub, Image, gpu, Volume, NetworkFileSystem
from helpers.data import format_dataset, score_prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import List
import pandas as pd

# GPU Configuration
gpu_config = gpu.A100()

# Finetuning Configuration ( Arrays are configurable parameters )
MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
]
DENSE_LAYER_DIMS = [128, 1000]
ACTIVATION_FUNCTIONS = [
    "tanh",
    "sigmoid",
]
SCHEDULER = [
    "constantlr",
    "warmupconstant",
    "warmuplinear",
    "warmupcosine",
    "warmupcosinewithhardrestarts",
]
DATASET_SIZE = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
WARMUP_STEPS = [500, 1000, 1500, 2000]
BATCH_SIZE = [32, 96]
MODEL_SAVE_PATH = "/output"
OPTIMIZATION_METRIC_FUNC = roc_auc_score
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-3
MAX_EPOCHS = 8
FREEZE_EMBEDDING_MODEL = [
    True,
    # False,
]

STUDY_NFS = NetworkFileSystem.new("modal-optimization")
JOURNAL_PATH = "/root/cache/journal.log"
STUDY_NAME = "optuna-optimization"

WORKERS = 5
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


@stub.function(
    image=image,
    gpu=gpu_config,
    network_file_systems={"/root/cache": STUDY_NFS},
    volumes={DATASET_DIR: DATASET_VOLUME},
    timeout=86400,
)
def objective(trial, existing_experiments: List[dict]):
    from sentence_transformers import SentenceTransformer, losses, evaluation, models
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    import torch.nn as nn
    import shutil
    import os

    # Delete the directory if it exists
    if os.path.exists(MODEL_SAVE_PATH):
        shutil.rmtree(MODEL_SAVE_PATH)

    activations = {
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    params = {
        "model_name": trial.suggest_categorical("model_name", MODELS),
        "dense_out_features": trial.suggest_int(
            "dense_out_features", *DENSE_LAYER_DIMS
        ),
        "activation_function": trial.suggest_categorical(
            "activation_function", ACTIVATION_FUNCTIONS
        ),
        "scheduler": trial.suggest_categorical("scheduler", SCHEDULER),
        "dataset_size": trial.suggest_categorical("dataset_size", DATASET_SIZE),
        "warmup_steps": trial.suggest_categorical("warmup_steps", WARMUP_STEPS),
        "learning_rate": trial.suggest_float(
            "learning_rate", MIN_LEARNING_RATE, MAX_LEARNING_RATE
        ),
        "freeze_embedding_model": trial.suggest_categorical(
            "freeze_embedding_model", FREEZE_EMBEDDING_MODEL
        ),
        "batch_size": trial.suggest_int("batch_size", *BATCH_SIZE),
    }
    # We load our model and freeze the layers ( see https://github.com/UKPLab/sentence-transformers/issues/680 )
    embedding_model = SentenceTransformer(params["model_name"])

    if params["freeze_embedding_model"]:
        auto_model = embedding_model._first_module().auto_model
        for param in auto_model.parameters():
            param.requires_grad = False

    dense_model = models.Dense(
        in_features=embedding_model.get_sentence_embedding_dimension(),
        out_features=params["dense_out_features"],
        # just make a dict of the params and then pass it to the nn module
        activation_function=activations[params["activation_function"]],
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
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=params["batch_size"]
    )

    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples, batch_size=params["batch_size"]
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        warmup_steps=params["warmup_steps"],
        scheduler=params["scheduler"],
        optimizer_params={
            "lr": params["learning_rate"],
        },
        save_best_model=True,
        show_progress_bar=True,
        epochs=MAX_EPOCHS,
        output_path=MODEL_SAVE_PATH,
    )

    # Why do we reload it? We don't even load the model
    # with the best weights?
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

    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(JOURNAL_PATH)
    )
    results = []
    # Set it so that we can resume a study in the event that something fails
    study = optuna.create_study(
        study_name="finetuning", storage=storage, load_if_exists=True
    )

    study.optimize(lambda trial: objective.remote(trial, results), n_trials=n_trials)

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


@stub.local_entrypoint()
def main():
    import time

    date = time.strftime("%Y-%m-%d")

    results = []

    for resp in optimize_hyperparameters.map(
        [NUM_TRIALS for _ in range(WORKERS)],
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(resp, Exception):
            print(f"Encountered Exception of {resp}")
            continue
        results.extend(resp)

    df = pd.DataFrame(results)
    df.to_csv(f"./paramsearch/{date}_optuna_trial_results.csv", index=False)

    df = get_dataframe.remote()
    df.to_csv(f"./paramsearch/{date}_optuna_journal_results.csv", index=False)
