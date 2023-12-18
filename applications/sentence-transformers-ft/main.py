from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import modal
import pathlib
from datetime import datetime
import os

model_id = "BAAI/bge-small-en-v1.5"
dataset_id = "quora"


# Functions for Modal Image build step
def download_model():
    SentenceTransformer(model_id)


def download_dataset():
    load_dataset(dataset_id, split="train")


# Modal constants
VOL_MOUNT_PATH = pathlib.Path("/vol")
GPU_CONFIG = "a10g"


# Modal resources
volume = modal.Volume.persisted(
    f"sentence-transformers-ft-{int(datetime.now().timestamp())}"
)
stub = modal.Stub("finetune-embeddings")
image = (
    modal.Image.debian_slim()
    .pip_install("sentence-transformers", "torch", "datasets")
    # we download the model and dataset to save them in the image in between runs
    .run_function(download_model)
    .run_function(download_dataset)
)

# Quora pairs dataset: https://huggingface.co/datasets/quora
dataset = load_dataset(dataset_id, split="train")
# Quora pairs dataset only contains a "train" split in huggingface, so we will manually split it into train and test
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=15000,
    volumes={VOL_MOUNT_PATH: volume},
    _allow_background_volume_commits=True,
)
def finetune():
    """
    Finetune a sentence transformer on the quora pairs dataset. Evaluates model performance before/after training

    Inspired by: https://github.com/UKPLab/sentence-transformers/blob/657da5fe23fe36058cbd9657aec6c7688260dd1f/examples/training/quora_duplicate_questions/training_MultipleNegativesRankingLoss.py
    """
    model = SentenceTransformer(model_id)

    train_examples = []
    for i in range(train_dataset.num_rows):
        text0 = train_dataset[i]["questions"]["text"][0]
        text1 = train_dataset[i]["questions"]["text"][1]
        is_duplicate = int(train_dataset[i]["is_duplicate"])
        train_examples.append(InputExample(texts=[text0, text1], label=is_duplicate))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.OnlineContrastiveLoss(model)

    test_examples = []
    for i in range(test_dataset.num_rows):
        text0 = test_dataset[i]["questions"]["text"][0]
        text1 = test_dataset[i]["questions"]["text"][1]
        is_duplicate = int(test_dataset[i]["is_duplicate"])
        test_examples.append(InputExample(texts=[text0, text1], label=is_duplicate))
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples,
    )

    # evaluator.name is used for how the file name is saved
    evaluator.csv_file = "binary_classification_evaluation_pre_train" + "_results.csv"
    pre_train_eval = evaluator(model, output_path=str(VOL_MOUNT_PATH))
    print("pre train eval score:", pre_train_eval)

    evaluator.csv_file = "binary_classification_evaluation" + "_results.csv"
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=10,
        output_path=str(VOL_MOUNT_PATH / f"{model_id.replace('/','--')}-ft"),
        checkpoint_path=str(VOL_MOUNT_PATH / f"checkpoints"),
        checkpoint_save_total_limit=5,
    )

    evaluator.csv_file = "binary_classification_evaluation_post_train" + "_results.csv"
    post_train_eval = evaluator(model, output_path=str(VOL_MOUNT_PATH))

    print("post train eval score:", post_train_eval)


# run on modal with `modal run main.py`
@stub.local_entrypoint()
def main():
    finetune.remote()


# run on local with `python main.py`
if __name__ == "__main__":
    VOL_MOUNT_PATH = pathlib.Path("./")
    finetune.local()
