from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import modal
import pathlib
from datetime import datetime

model_id = "BAAI/bge-small-en-v1.5"
dataset_id = "quora"


# Functions for Modal Image build step
def download_model():
    SentenceTransformer(model_id)


def download_dataset():
    load_dataset(dataset_id, split="train")


# Modal constants
VOL_MOUNT_PATH = pathlib.Path("/vol")
GPU_CONFIG = "A10G"


# Modal resources
volume = modal.Volume.persisted("sentence-transformers-ft")
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


def get_quora_examples(split):
    """
    Return a list of InputExample for quora dataset based on the split

    :param split: dataset split
    """
    # ms marco huggingface dataset: https://huggingface.co/datasets/ms_marco
    if split not in ["train", "test"]:
        raise ValueError("split must be in [train, test]")

    examples = []
    dataset_split = train_test_split[split]
    # for agility, train with smaller dataset
    # TODO: for actual training use the full dataset
    n_examples = dataset_split.num_rows // 20
    # n_examples = dataset_split.num_rows

    # make dataset only have positives pairs
    for i in range(n_examples):
        data = dataset_split[i]
        text0 = data["questions"]["text"][0]
        text1 = data["questions"]["text"][1]

        if split == "train":
            # since we are using MultipleNegativesRankingLoss, only add positive pairs
            if data["is_duplicate"]:
                examples.append(InputExample(texts=[text0, text1], label=1))
                # if A is a duplicate of B, then B is a duplicate of A
                examples.append(InputExample(texts=[text1, text0], label=1))
        else:
            examples.append(
                InputExample(texts=[text0, text1], label=int(data["is_duplicate"]))
            )

    return examples


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

    train_examples = get_quora_examples("train")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    test_examples = get_quora_examples("test")

    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples
    )

    pre_train_eval = evaluator.compute_metrices(model)
    print("pre train eval", pre_train_eval)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=10,
        output_path=str(
            VOL_MOUNT_PATH
            / f"{model_id.replace('/','--')}-ft-{datetime.now().timestamp()}"
        ),
        checkpoint_path=str(
            VOL_MOUNT_PATH / f"checkpoints-{datetime.now().timestamp()}"
        ),
        checkpoint_save_total_limit=5,
    )

    post_train_eval = evaluator.compute_metrices(model)

    print("post train eval", post_train_eval)


# run on modal with `modal run main.py`
@stub.local_entrypoint()
def main():
    finetune.remote()


# run on local with `python main.py`
if __name__ == "__main__":
    finetune.local()
