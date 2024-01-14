from modal import Image, Stub, Volume, Secret, gpu
import os

# Model Configuration
MODEL_ID = "BAAI/bge-base-en-v1.5"
GPU_CONFIG = gpu.A100()


# Dataset Configuration
DATASET_NAME = "quora"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.persisted("datasets")

# Test Configuration
TEST_PERCENTAGE = 0.1

stub = Stub("cohere-embeddings")


def download_model():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_ID)


image = (
    Image.debian_slim()
    .env({"MODEL_ID": MODEL_ID})
    .pip_install(
        "cohere", "datasets", "sentence-transformers", "scikit-learn", "tabulate"
    )
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
    volumes={DATASET_DIR: DATASET_VOLUME},
    secret=Secret.from_name("cohere"),
)
def benchmark_cohere():
    from cohere import Client
    from cohere.responses import Embeddings
    from datasets import load_from_disk
    from sentence_transformers import util
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from tabulate import tabulate

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    train_test_split = dataset.train_test_split(test_size=TEST_PERCENTAGE, seed=42)
    test_dataset = train_test_split["test"].select(range(400))
    co = Client(os.environ["COHERE_API_KEY"])

    thresholds = [0.3, 0.5, 0.7, 0.9]

    computed_values = []
    for input_type in [
        "search_document",
        "search_query",
        "classification",
        "clustering",
    ]:
        predictions = []
        actual_label = []
        values = []
        for row in test_dataset:
            response: Embeddings = co.embed(
                texts=row["questions"]["text"],
                model="embed-english-v3.0",
                input_type=input_type,
            )
            e1, e2 = response.embeddings
            cosine_scores = util.cos_sim(e1, e2)
            predictions.append(cosine_scores)
            actual_label.append(1 if row["is_duplicate"] else 0)

        for threshold in thresholds:
            pred_label = [1 if i >= threshold else 0 for i in predictions]
            accuracy = accuracy_score(actual_label, pred_label)
            precision = precision_score(actual_label, pred_label)
            recall = recall_score(actual_label, pred_label)
            values.append([threshold, accuracy, recall, precision])
        computed_values.append(values)

    headers = ["Threshold", "Accuracy", "Precision", "Recall"]
    print(
        tabulate(
            [
                [
                    tabulate(computed_values[0], headers, tablefmt="heavy_outline"),
                    tabulate(computed_values[1], headers, tablefmt="heavy_outline"),
                ]
            ],
            ["Search Document", "Search Query"],
            tablefmt="simple",
        )
    )
    print(
        tabulate(
            [
                [
                    tabulate(computed_values[2], headers, tablefmt="heavy_outline"),
                    tabulate(computed_values[3], headers, tablefmt="heavy_outline"),
                ]
            ],
            ["Classification", "Clustering"],
            tablefmt="simple",
        ),
    )


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME}, gpu=GPU_CONFIG)
def benchmark_mteb_model():
    from cohere import Client
    from cohere.responses import Embeddings
    from datasets import load_from_disk
    from sentence_transformers import util, SentenceTransformer
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from tabulate import tabulate

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    train_test_split = dataset.train_test_split(test_size=TEST_PERCENTAGE, seed=42)
    test_dataset = train_test_split["test"].select(range(400))

    model = SentenceTransformer(MODEL_ID)

    predictions = []
    actual_label = []
    values = []
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for row in test_dataset:
        # Compute embedding for both lists
        s1, s2 = row["questions"]["text"]

        e1 = model.encode(s1, convert_to_tensor=True)
        e2 = model.encode(s2, convert_to_tensor=True)

        cosine_scores = util.cos_sim(e1, e2)
        predictions.append(cosine_scores)
        actual_label.append(1 if row["is_duplicate"] else 0)

    for threshold in thresholds:
        pred_label = [1 if i >= threshold else 0 for i in predictions]
        accuracy = accuracy_score(actual_label, pred_label)
        precision = precision_score(actual_label, pred_label)
        recall = recall_score(actual_label, pred_label)
        values.append([threshold, accuracy, recall, precision])

    print(
        tabulate(
            [
                [
                    tabulate(
                        values,
                        ["Threshold", "Accuracy", "Precision", "Recall"],
                        tablefmt="heavy_outline",
                    )
                ]
            ],
            [MODEL_ID],
            tablefmt="simple",
        )
    )


@stub.local_entrypoint()
def main():
    benchmark_mteb_model.remote()
