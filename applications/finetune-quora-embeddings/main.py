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
TEST_PERCENTAGE = 0.9
BATCH_SIZE = 32
EPOCHS = 3
SCHEDULER = "warmuplinear"
CHECKPOINT_VOLUME = Volume.persisted("checkpoints")
CHECKPOINT_DIR = "/checkpoints"
SAVE_CHECKPOINT = True
MAX_CHECKPOINTS_SAVED = 3

# Wandb Configuration
WANDB_PROJECT_NAME = "quora-finetuning"

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


def generate_quora_input_example(examples):
    from sentence_transformers import InputExample

    return [
        InputExample(
            texts=[
                example["questions"]["text"][0],
                example["questions"]["text"][1],
            ],
            label=int(example["is_duplicate"]),
        )
        for example in examples
    ]


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={DATASET_DIR: DATASET_VOLUME, CHECKPOINT_DIR: CHECKPOINT_VOLUME},
    timeout=86400,
    secret=Secret.from_name("wandb"),
)
def finetune_model():
    from sentence_transformers import SentenceTransformer, models, losses, evaluation
    from datasets import load_from_disk
    from torch import nn
    from torch.utils.data import DataLoader
    import wandb

    # Initialise Wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project=WANDB_PROJECT_NAME,
        # track hyperparameters and run metadata
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "scheduler": SCHEDULER,
            "test_percentage": TEST_PERCENTAGE,
        },
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
    train_test_split = dataset.train_test_split(test_size=TEST_PERCENTAGE)
    train_dataset = generate_quora_input_example(train_test_split["train"])
    test_dataset = generate_quora_input_example(train_test_split["test"])

    # Define the Key components
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.OnlineContrastiveLoss(model)
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_dataset,
    )

    print("### Model Evaluation Without Training ###")
    pre_train_eval = evaluator(model)
    print(
        f"Pre train eval score (highest Average Precision across all distance functions):{pre_train_eval}"
    )
    wandb.log({"score": pre_train_eval})

    def log_metrics(score, epoch, steps):
        print(f"Epoch {epoch}: score {score}")
        wandb.log({"score": score})
        if SAVE_CHECKPOINT:
            CHECKPOINT_VOLUME.commit()

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        output_path=f"{CHECKPOINT_DIR}/{wandb.run.name}",
        checkpoint_path=f"{CHECKPOINT_DIR}/{wandb.run.name}/checkpoints/"
        if SAVE_CHECKPOINT
        else None,
        checkpoint_save_total_limit=3,
        scheduler=SCHEDULER,
        callback=log_metrics,
    )

    print("### Model Evaluation Without Training ###")
    post_train_eval = evaluator(model)
    print(
        f"post_train_eval score (highest Average Precision across all distance functions):{post_train_eval}"
    )


@stub.local_entrypoint()
def main():
    download_dataset.remote()
    finetune_model.remote()
