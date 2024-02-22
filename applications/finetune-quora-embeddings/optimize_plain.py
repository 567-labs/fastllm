from dataclasses import dataclass
import json
from modal import Stub, Image, gpu, Volume, NetworkFileSystem
from helpers.data import format_dataset, score_prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd

# GPU Configuration
gpu_config = gpu.A10G()

# Finetuning Configuration ( Arrays are configurable parameters )
MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    # "BAAI/bge-large-en-v1.5",
]
SCHEDULER = [
    "constantlr",
    "warmupconstant",
    "warmuplinear",
    "warmupcosine",
    "warmupcosinewithhardrestarts",
]
DATASET_SIZE = [2000, 4000, 8000, 16000]  # , 32000, 64000, 128000]
WARMUP_STEPS = [500, 1000, 1500]
BATCH_SIZE = [32, 64]
MODEL_SAVE_PATH = "/output"
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-3
MAX_EPOCHS = 3
FREEZE_EMBEDDING_MODEL = [
    True,
    False,
]

STUDY_NFS = NetworkFileSystem.new("modal-optimization")
JOURNAL_PATH = "/root/cache/journal.log"
STUDY_NAME = "optuna-optimization"

# DATASET CONFIG
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.persisted("datasets")
CACHE_DIRECTORY = f"{DATASET_DIR}/cached-embeddings"
TEST_SET_SIZE = 10000

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


@dataclass
class ModelConfig:
    model_name: str
    dataset_size: int
    dense_out_features: int
    learning_rate: float
    scheduler: str
    warmup_steps: int
    freeze_embedding_model: bool
    batch_size: int
    num_epochs: int


def random_search_config(model_name, dataset_size):
    """
    Randomly sample from the configuration space
    """
    import random

    scheduler = random.choice(SCHEDULER)
    warmup_steps = random.choice(WARMUP_STEPS)
    batch_size = random.choice(BATCH_SIZE)
    freeze_embedding_model = random.choice(FREEZE_EMBEDDING_MODEL)
    learning_rate = random.uniform(MIN_LEARNING_RATE, MAX_LEARNING_RATE)
    dense_out_features = random.randint(100, 1000)
    num_epochs = MAX_EPOCHS  # This could also be made configurable if desired

    return ModelConfig(
        model_name=model_name,
        dataset_size=dataset_size,
        dense_out_features=dense_out_features,
        learning_rate=learning_rate,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        freeze_embedding_model=freeze_embedding_model,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


@stub.function(
    image=image,
    gpu=gpu_config,
    network_file_systems={"/root/cache": STUDY_NFS},
    volumes={DATASET_DIR: DATASET_VOLUME},
    timeout=86400,
)
def objective(
    config: ModelConfig,
):
    from sentence_transformers import SentenceTransformer, losses, evaluation, models
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    import torch.nn as nn
    import shutil
    import os

    model_name = config.model_name
    dataset_size = config.dataset_size
    dense_out_features = config.dense_out_features
    learning_rate = config.learning_rate
    scheduler = config.scheduler
    warmup_steps = config.warmup_steps
    freeze_embedding_model = config.freeze_embedding_model
    batch_size = config.batch_size
    num_epochs = config.num_epochs

    print(f"Training model {model_name} {config}")

    # Load the model
    embedding_model = SentenceTransformer(model_name)

    # Delete the directory if it exists
    if os.path.exists(MODEL_SAVE_PATH):
        shutil.rmtree(MODEL_SAVE_PATH)

    # Model configuration
    dim_emb = embedding_model.get_sentence_embedding_dimension()

    # Freeze the embedding model
    if freeze_embedding_model:
        for param in embedding_model._first_module().auto_model.parameters():
            param.requires_grad = False

    # Define the model architecture with additional dense layer
    dense_model = models.Dense(
        in_features=dim_emb,
        out_features=dense_out_features,
        activation_function=nn.Tanh(),
    )
    pooling_model = models.Pooling(dim_emb)

    # Initialize the model
    model = SentenceTransformer(
        modules=[embedding_model, pooling_model, dense_model], device="cuda"
    )

    # Load the dataset
    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    train_dataset = dataset["train"].select(range(dataset_size))
    test_dataset = dataset["test"].select(range(TEST_SET_SIZE))

    # Format the dataset
    train_examples, test_examples = (
        format_dataset(train_dataset),
        format_dataset(test_dataset),
    )

    # Create dataloaders and evaluator
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples, batch_size=batch_size
    )
    train_loss = losses.OnlineContrastiveLoss(model)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        optimizer_params={"lr": learning_rate},
        save_best_model=True,
        epochs=num_epochs,
        output_path=MODEL_SAVE_PATH,
    )

    # Reload the best model
    model = SentenceTransformer(MODEL_SAVE_PATH)

    # Score and evaluate the model
    predictions, test_labels = score_prediction(model, train_dataset, test_dataset)
    eval_results = {
        f"metric_{metric}": round(function(test_labels, predictions), 4)
        for metric, function in METRICS.items()
    }
    eval_results["model_name"] = model_name
    eval_results["dataset_size"] = dataset_size
    eval_results["dense_out_features"] = dense_out_features
    eval_results["learning_rate"] = learning_rate
    eval_results["scheduler"] = scheduler
    eval_results["warmup_steps"] = warmup_steps
    eval_results["freeze_embedding_model"] = freeze_embedding_model
    eval_results["batch_size"] = batch_size
    eval_results["num_epochs"] = num_epochs
    return eval_results


def generate_configs(n_trials):
    for model in MODELS:
        for dataset_size in DATASET_SIZE:
            for _ in range(n_trials):
                yield random_search_config(model, dataset_size)


@stub.local_entrypoint()
def main():
    import time

    date = time.strftime("%Y-%m-%d-%H-%M")

    results = []

    for experiment_result in objective.map(generate_configs(n_trials=3)):
        if isinstance(experiment_result, Exception):
            print(f"Encountered Exception of {experiment_result}")
            continue
        results.append(experiment_result)

    df = pd.DataFrame(results)
    df.to_csv(f"./paramsearch/{date}_plain_trial_results.csv", index=False)
