from cgi import test
import json
from modal import Stub, Image, gpu, Volume, NetworkFileSystem
from regex import F
from helpers.data import format_dataset, score_prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import List
import pandas as pd

# GPU Configuration
gpu_config = gpu.A100()

# Finetuning Configuration ( Arrays are configurable parameters )
MODELS = [
    "BAAI/bge-base-en-v1.5",
    # "BAAI/bge-large-en-v1.5",
]
DENSE_LAYER_DIMS = [128, 512]
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
DATASET_SIZE = [
    2000,
]  # 4000, 8000, 16000, 32000, 64000, 128000]
WARMUP_STEPS = [500, 1000, 1500, 2000]
BATCH_SIZE = [18, 32, 64, 96]
MODEL_SAVE_PATH = "/output"
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-3
MAX_EPOCHS = 5
FREEZE_EMBEDDING_MODEL = [
    # True,
    False,
]

STUDY_NFS = NetworkFileSystem.new("modal-optimization")
JOURNAL_PATH = "/root/cache/journal.log"
STUDY_NAME = "optuna-optimization"

WORKERS = 5
NUM_TRIALS = 10

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


def objective(model_name: str, dataset_size: int, trial):
    from sentence_transformers import SentenceTransformer, losses, evaluation, models
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    import torch.nn as nn
    import shutil
    import os

    # Load the model
    embedding_model = SentenceTransformer(model_name)

    # Delete the directory if it exists
    if os.path.exists(MODEL_SAVE_PATH):
        shutil.rmtree(MODEL_SAVE_PATH)

    # Define activations
    activations = {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}

    # Suggest parameters
    lr = trial.suggest_float("learning_rate", MIN_LEARNING_RATE, MAX_LEARNING_RATE)
    dense_out_features = trial.suggest_int("dense_out_features", *DENSE_LAYER_DIMS)
    activation_function_key = trial.suggest_categorical(
        "activation_function", ACTIVATION_FUNCTIONS
    )
    activation_function = activations[activation_function_key]
    scheduler = trial.suggest_categorical("scheduler", SCHEDULER)
    warmup_steps = trial.suggest_categorical("warmup_steps", WARMUP_STEPS)
    freeze_embedding_model = trial.suggest_categorical(
        "freeze_embedding_model", FREEZE_EMBEDDING_MODEL
    )
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE)
    num_epochs = MAX_EPOCHS

    # Freeze the embedding model
    if freeze_embedding_model:
        print("Freezing the embedding model")
        auto_model = embedding_model._first_module().auto_model
        for param in auto_model.parameters():
            param.requires_grad = False

    # Define the model architecture with additional dense layer
    dense_model = models.Dense(
        in_features=embedding_model.get_sentence_embedding_dimension(),
        out_features=dense_out_features,
        activation_function=activation_function,
    )

    # Initialize the model
    model = SentenceTransformer(modules=[embedding_model, dense_model])

    # Load the dataset
    print(f"Loading dataset with {dataset_size} samples")
    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    train_dataset = dataset["train"].select(range(dataset_size))
    test_dataset = dataset["test"]

    # Format the dataset
    train_examples, test_examples = (
        format_dataset(train_dataset),
        format_dataset(test_dataset),
    )

    print("Creating the dataloaders")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    print("Creating the evaluator & loss function")
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples, batch_size=batch_size
    )

    train_loss = losses.OnlineContrastiveLoss(model)

    print("Training the model")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        optimizer_params={"lr": lr},
        save_best_model=True,
        show_progress_bar=True,
        epochs=num_epochs,
        output_path=MODEL_SAVE_PATH,
    )

    # Reload the model with the best weights
    model = SentenceTransformer(MODEL_SAVE_PATH)

    # Score the model
    print("Scoring the model")
    predictions, test_labels = score_prediction(model, train_dataset, test_dataset)

    # Evaluate the model
    eval_results = {
        metric: round(function(test_labels, predictions), 4)
        for metric, function in METRICS.items()
    }
    print(f"Eval results: {eval_results}")

    # Return the objective to minimize
    return 1 - accuracy_score(test_labels, predictions)


@stub.function(
    image=image,
    gpu=gpu_config,
    network_file_systems={"/root/cache": STUDY_NFS},
    volumes={DATASET_DIR: DATASET_VOLUME},
    timeout=86400,
)
def optimize_hyperparameters(model_name: str, dataset_size: int, n_trials: int):
    import optuna

    # Set it so that we can resume a study in the event that something fails
    # storage = optuna.storages.JournalStorage(
    #    optuna.storages.JournalFileStorage(JOURNAL_PATH)
    # )
    study = optuna.create_study(
        study_name="finetuning",  # storage=storage,load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(model_name, dataset_size, trial),
        n_trials=n_trials,
    )

    best_params = study.best_params
    best_params["model_name"] = model_name
    best_params["dataset_size"] = dataset_size
    return best_params


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

    for response in optimize_hyperparameters.starmap(
        [
            (model, dataset_size, NUM_TRIALS)
            for model in MODELS
            for dataset_size in DATASET_SIZE
        ],
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(response, Exception):
            print(f"Encountered Exception of {response}")
            continue
        best_params = response
        results.append(best_params)

    df = pd.DataFrame(results)
    df.to_csv(f"./paramsearch/{date}_optuna_trial_results.csv", index=False)

    df = get_dataframe.remote()
    df.to_csv(f"./paramsearch/{date}_optuna_journal_results.csv", index=False)
