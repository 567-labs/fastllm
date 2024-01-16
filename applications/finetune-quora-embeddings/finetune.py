from modal import Image, gpu, Stub, Volume, Secret


# Dataset Configuration
DATASET_NAME = "quora"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.persisted("datasets")

# Stub Configuration
MODEL_ID = "BAAI/bge-base-en-v1.5"
GPU_CONFIG = gpu.A100()

# Finetune Configuration

# Wandb Configuration
WANDB_PROJECT_NAME = "quora-finetuning"
ENABLE_LOGGING = True

stub = Stub("finetune")


def download_model_and_dataset():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_ID)


image = (
    Image.debian_slim()
    .pip_install(
        "sentence-transformers",
        "torch",
        "datasets",
        "scikit-learn",
        "huggingface_hub",
        "hf_transfer",
        "wandb",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_and_dataset)
)


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
    timeout=86400,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secrets=[Secret.from_name("huggingface-credentials"), Secret.from_name("wandb")],
)
def train_model(train_dataset_percentage: float):
    from sentence_transformers import SentenceTransformer, losses, evaluation, util
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from datasets import load_from_disk
    import math
    import time
    import numpy as np
    import wandb

    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer(MODEL_ID)

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    val_dataset = dataset["val"]
    train_dataset = dataset["train"]

    if ENABLE_LOGGING:
        # Initialise Wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project=WANDB_PROJECT_NAME,
            # track hyperparameters and run metadata
            group=f"{MODEL_ID}-{train_dataset_percentage}",
        )

    dataset_slice = math.floor(len(train_dataset) * train_dataset_percentage)
    train_dataset = generate_quora_input_example(
        train_dataset.select(range(dataset_slice))
    )
    val_dataset = generate_quora_input_example(val_dataset)

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    train_loss = losses.OnlineContrastiveLoss(model)
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        val_dataset,
    )

    def log_metrics(score, epoch, steps):
        print(f"Epoch {epoch}: score {score}")

        if ENABLE_LOGGING:
            wandb.log({"score": score})

    loss = evaluator(model)

    for i in range(6):
        # Tune the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
            callback=log_metrics,
        )
        new_loss = evaluator(model)
        if ENABLE_LOGGING:
            wandb.log({"loss": loss})

        if abs(new_loss - loss) < 0.01:
            break

    start = time.time()

    # Process texts in batches
    texts1 = [row.texts[0] for row in val_dataset]
    texts2 = [row.texts[1] for row in val_dataset]
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    optimized_predictions = np.diag(cosine_scores.cpu()).tolist()
    optimized_labels = [1 if row.label else 0 for row in val_dataset]

    print(f"Generated calculations in {time.time()-start}s")

    auc = roc_auc_score(optimized_labels, optimized_predictions)

    if ENABLE_LOGGING:
        wandb.log({"auc": auc})

    DATASET_HF_UPLOAD_REPO_NAME = (
        f"567-labs/{MODEL_ID.split('/').pop()}-ft-quora-{train_dataset_percentage}"
    )
    path = f"/checkpoints/{DATASET_HF_UPLOAD_REPO_NAME}"
    model.save(path)
    from huggingface_hub import HfApi
    import os

    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        private=False,
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=path,
        repo_id=DATASET_HF_UPLOAD_REPO_NAME,
        repo_type="model",
        multi_commits=True,
        multi_commits_verbose=True,
    )

    return auc


@stub.local_entrypoint()
def main():
    from tabulate import tabulate
    import matplotlib.pyplot as plt

    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
    res = {}
    for percentage, auc in zip(
        percentages, train_model.map(percentages, order_outputs=True)
    ):
        res[percentage] = auc

    for percentage in percentages:
        print(res[percentage])

    values = [[model, auc] for model, auc in res.items()]
    values.sort(key=lambda x: x[1], reverse=True)
    print(tabulate(values, ["Data Percentage", "AUC"]))

    x_values = list(res.keys())
    y_values = list(res.values())

    plt.plot(
        x_values, y_values, marker="o"
    )  # 'o' creates a circle marker at each data point
    plt.xlim(0, 1)  # Set the limit of x-axis from 0 to 1
    plt.xlabel("Data Percentage")  # Label for x-axis
    plt.ylabel("AUC Score")  # Label for y-axis
    plt.title("Data Percentage vs AUC Performance")
    plt.text(
        1, 0.897, "Cohere Embedding Model (0.89)", va="center", ha="right"
    )  # Position the text at the end of the line
    plt.axhline(
        y=0.891776,
        color="r",
        linestyle="--",
        label="embed-multilingual-v3.0 ( Cohere ) AUC Performance",
    )  # Add the horizontal line
    plt.savefig("res.png")
