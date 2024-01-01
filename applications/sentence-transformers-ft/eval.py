from datasets import load_dataset
import modal
from modal import Image
from sentence_transformers import InputExample, evaluation, SentenceTransformer
import csv

DATASET_ID = "quora"
GPU_CONFIG = "a10g"
MODEL_IDS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-small",
    "intfloat/e5-small-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/multi-qa-distilbert-cos-v1",
    "llmrails/ember-v1",
    "krunchykat/bge-base-en-v1.5-ft-quora",
]

stub = modal.Stub("embeddings-eval")


def download_dataset():
    # download the dataset in the image to cache it across multiple containers
    load_dataset(DATASET_ID, split="train")


image = (
    Image.debian_slim()
    .pip_install("datasets", "sentence-transformers")
    .run_function(download_dataset)
)


@stub.function(image=image, gpu=GPU_CONFIG, timeout=1000, concurrency_limit=10)
def eval(model_id: str):
    # Quora pairs dataset: https://huggingface.co/datasets/quora
    # Quora pairs dataset only contains a "train" split in huggingface
    dataset = load_dataset(DATASET_ID, split="train")

    # Only evaluate on test set, use the same split seed used to train your finetuned model if applicable
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = train_test_split["test"]
    test_examples = [
        InputExample(
            texts=[
                test_dataset[i]["questions"]["text"][0],
                test_dataset[i]["questions"]["text"][1],
            ],
            label=int(test_dataset[i]["is_duplicate"]),
        )
        for i in range(test_dataset.num_rows)
    ]
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples, show_progress_bar=True
    )

    model = SentenceTransformer(model_id)

    metrices = evaluator.compute_metrices(model)
    print(f"Metrices for {model_id}\n{metrices}")

    return metrices


@stub.local_entrypoint()
def main():
    def flatten_dict(d):
        def expand(key, value):
            if isinstance(value, dict):
                return [(key + "_" + k, v) for k, v in flatten_dict(value).items()]
            else:
                return [(key, value)]

        items = [item for k, v in d.items() for item in expand(k, v)]
        return dict(items)

    # run evals in parallel on model
    metrices_list = eval.map(MODEL_IDS)

    # Open the CSV file once, outside the loop
    with open("eval_metrics.csv", "w", newline="") as csvfile:
        # Initialize the CSV writer
        writer = None

        for model_id, metrices in zip(MODEL_IDS, metrices_list):
            metrices_flattened = flatten_dict(metrices)
            metrices_flattened["model_id"] = model_id

            # Manually set the order of fieldnames so "model_id" is first
            fieldnames = ["model_id"] + [
                key for key in metrices_flattened.keys() if key != "model_id"
            ]

            # Initialize the writer with fieldnames if it's not already set
            if writer is None:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

            # Write the row
            writer.writerow(metrices_flattened)
