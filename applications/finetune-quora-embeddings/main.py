from lib.data import generate_train_test_dataset, generate_predictions
from lib.stats import generate_scores, log_metrics
from modal import Image, gpu, Stub, Volume, Secret
import os

# Stub Configuration
MODEL_ID = "BAAI/bge-base-en-v1.5"
GPU_CONFIG = gpu.A100()

# Dataset Configuration
DATASET_NAME = "quora"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.persisted("datasets")

# Embedding Configuration
INCLUDE_DENSE_LAYER = True
DENSE_OUT_FEATURES = 768

# Finetune Configuration
TEST_PERCENTAGE = 0.001
TRAIN_PERCENTAGE = 0.9
BATCH_SIZE = 32
EPOCHS = 1
SCHEDULER = "warmuplinear"
CHECKPOINT_VOLUME = Volume.persisted("checkpoints")
CHECKPOINT_DIR = "/checkpoints"
SAVE_CHECKPOINT = False
MAX_CHECKPOINTS_SAVED = 3

# Wandb Configuration
WANDB_PROJECT_NAME = "quora-finetuning"
ENABLE_LOGGING = False

# Evaluation Configuration
THRESHOLD = 0.5

stub = Stub("finetune-quora-embeddings")


def download_model():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_ID)


requirements_txt = os.path.join(os.path.dirname(__file__), "requirements.txt")
image = (
    Image.debian_slim()
    .env({"MODEL_ID": MODEL_ID})
    .pip_install_from_requirements(requirements_txt)
    .run_function(download_model)
)


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME})
def download_dataset():
    from datasets import load_dataset

    dataset_path = f"{DATASET_DIR}/{DATASET_NAME}"

    if os.path.exists(dataset_path):
        return

    dataset = load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, num_proc=6
    )
    dataset.save_to_disk(dataset_path)
    DATASET_VOLUME.commit()


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={DATASET_DIR: DATASET_VOLUME, CHECKPOINT_DIR: CHECKPOINT_VOLUME},
    timeout=86400,
    secret=Secret.from_name("wandb"),
    interactive=True,
)
def finetune_model(holdout_percentage: float = 0.9):
    from sentence_transformers import (
        SentenceTransformer,
        models,
        losses,
        evaluation,
    )
    from datasets import load_from_disk
    from torch import nn
    from torch.utils.data import DataLoader
    import wandb

    if ENABLE_LOGGING:
        # Initialise Wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"{WANDB_PROJECT_NAME}",
            # track hyperparameters and run metadata
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "scheduler": SCHEDULER,
                "test_percentage": holdout_percentage,
            },
            group=f"{WANDB_PROJECT_NAME}-{holdout_percentage}",
        )

    # Validate that sentence tranformer works
    model = SentenceTransformer(MODEL_ID)
    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")

    if INCLUDE_DENSE_LAYER:
        dense_layer = models.Dense(
            in_features=model.get_sentence_embedding_dimension(),
            out_features=DENSE_OUT_FEATURES,
            activation_function=nn.Tanh(),
        )
        model = SentenceTransformer(modules=[model, dense_layer])

    model.to("cuda")

    # For simplicity we only split into 2 datasets, but you can add another split for "eval", 3 splits in total, if desired
    train_dataset, test_dataset = generate_train_test_dataset(
        dataset, TEST_PERCENTAGE, holdout_percentage
    )

    # Define the Key components
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.OnlineContrastiveLoss(model)
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_dataset,
    )

    # TODO log wandb here

    print("### Model Evaluation Without Training ###")

    print(
        "Pre train eval score (highest Average Precision across all distance functions)"
    )

    predictions = generate_predictions(test_dataset, model)
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        print(f"-----Threshold: {threshold} ----------")
        precision, recall, accuracy = generate_scores(
            predictions, test_dataset, threshold=threshold
        )
        print("------\n")

    current_loss = float("inf")
    for i in range(10):
        print(f"Starting Iteration {i}")
        predictions = generate_predictions(test_dataset, model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=2,
            # output_path=f"{CHECKPOINT_DIR}/{wandb.run.name}",
            # checkpoint_path=f"{CHECKPOINT_DIR}/{wandb.run.name}/checkpoints/"
            # if SAVE_CHECKPOINT
            # else None,
            # checkpoint_save_total_limit=3,
            scheduler=SCHEDULER,
            callback=log_metrics,
        )
        print("Model Finished Training")
        # We log the highest accuracy, precision and recal
        prec = 0
        final_vals = []
        chosen_threshold = 0.3
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            print(f"-----Threshold: {threshold} ----------")
            precision, recall, accuracy = generate_scores(
                predictions, test_dataset, threshold=threshold, print_scores=False
            )
            if precision > prec:
                chosen_threshold = threshold
                final_vals = [precision, recall, accuracy]
                prec = precision
            print("------\n")

        if ENABLE_LOGGING:
            precision, recall, accuracy = final_vals
            wandb.log(
                {
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "threshold": chosen_threshold,
                }
            )

        loss = evaluator(model)
        if abs(current_loss - loss) < 0.1:
            break
        current_loss = loss
        print(f"Current Loss is {loss}")


@stub.local_entrypoint()
def main():
    download_dataset.remote()
    finetune_model.remote(0.3)
