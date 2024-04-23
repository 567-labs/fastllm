from dataclasses import dataclass
from itertools import product
import json
from modal import Stub, Image, gpu, Volume, NetworkFileSystem
from helpers.data import format_dataset, score_prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import os

# GPU Configuration
gpu_config = gpu.A10G()

# Finetuning Configuration ( Arrays are configurable parameters )
MODELS = [
    "jinaai/jina-embeddings-v2-small-en",
    "all-MiniLM-L12-v2",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-mpnet-base-v2",
]
DATASET_SIZE = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
DENSE_OUT_FEATURES = [256, 512]
SCHEDULER = ["warmuplinear"]
WARMUP_STEPS = [500]
FREEZE_EMBEDDING_MODEL = [True]
BATCH_SIZE = [32]
MAX_EPOCHS = 8

# DATASET CONFIG
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.from_name("modal-optimization", create_if_missing=True)
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
    .pip_install("sentence-transformers", "torch", "datasets")
    .run_function(download_model)
)


@dataclass(frozen=True)
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


def random_search_config(model_name, dataset_size, freeze_embedding_model):
    """
    Randomly sample from the configuration space
    """
    import random

    scheduler = random.choice(SCHEDULER)
    warmup_steps = random.choice(WARMUP_STEPS)
    batch_size = random.choice(BATCH_SIZE)
    dense_out_features = random.choice(DENSE_OUT_FEATURES)
    num_epochs = MAX_EPOCHS  # This could also be made configurable if desired

    return ModelConfig(
        model_name=model_name,
        dataset_size=dataset_size,
        freeze_embedding_model=freeze_embedding_model,
        dense_out_features=dense_out_features,
        learning_rate=0.0001,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


@stub.function(
    image=image,
    gpu=gpu_config,
    volumes={DATASET_DIR: DATASET_VOLUME},
    concurrency_limit=50,
    allow_concurrent_inputs=True,
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
    model_config_hash = hash(config)
    MODEL_SAVE_PATH = f"/output/{model_config_hash}"

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

    print(json.dumps(eval_results, indent=2))
    return eval_results


def generate_configs(n_trials):
    configs = set()
    for model, sample_size, freeze_embedding_model in product(
        MODELS, DATASET_SIZE, FREEZE_EMBEDDING_MODEL
    ):
        for _ in range(n_trials):
            config = random_search_config(model, sample_size, freeze_embedding_model)
            config_hash = hash(config)
            if config_hash not in configs:
                yield config
                configs.add(config_hash)


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME})
def download_dataset():
    from datasets import load_dataset

    dataset_path = f"{DATASET_DIR}/{DATASET_NAME}"

    if os.path.exists(dataset_path):
        print("Dataset Exists")
        return

    dataset = load_dataset(DATASET_NAME)

    dataset.save_to_disk(dataset_path)
    DATASET_VOLUME.commit()


@stub.local_entrypoint()
def main():
    import time

    date = time.strftime("%Y-%m-%d-%H-%M")

    download_dataset.remote()

    results = []
    for experiment_result in objective.map(
        generate_configs(n_trials=1), return_exceptions=True
    ):
        if isinstance(experiment_result, Exception):
            print(f"Encountered Exception of {experiment_result}")
            continue
        results.append(experiment_result)
        # dumb but... save the results to a file every time a new result is available
        # This is to ensure that the results are not lost if the job is interrupted
        df = pd.DataFrame(results).sort_values("metric_accuracy", ascending=False)
        df.to_csv(f"./paramsearch/{date}_plain_trial_results.csv", index=False)

        # Save the results to a markdown file, this is useful for viewing the results in a human readable format
        with open(f"./paramsearch/{date}_plain_trial_results.md", "w") as f:
            f.write(df.to_markdown())
