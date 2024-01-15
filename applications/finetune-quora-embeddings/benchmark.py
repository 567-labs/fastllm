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
MAXIMUM_ELEMENTS_TO_TEST = 300
COHERE_MODEL = "embed-english-v3.0"

stub = Stub("cohere-embeddings")


image = Image.debian_slim().pip_install(
    "cohere", "datasets", "sentence-transformers", "scikit-learn", "tabulate"
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


@stub.function(
    image=image, volumes={DATASET_DIR: DATASET_VOLUME}, gpu=GPU_CONFIG, timeout=1200
)
def benchmark_mteb_model(model_name):
    from datasets import load_from_disk
    from sentence_transformers import util, SentenceTransformer
    from sklearn.metrics import roc_auc_score

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    train_test_split = dataset.train_test_split(test_size=TEST_PERCENTAGE, seed=42)
    test_dataset = train_test_split["test"].select(range(MAXIMUM_ELEMENTS_TO_TEST))

    model = SentenceTransformer(model_name)

    predictions = []
    actual_label = []

    for idx, row in enumerate(test_dataset):
        if idx % 100 == 0:
            print(f"Model: {model_name}, Iteration: {idx}/{MAXIMUM_ELEMENTS_TO_TEST}")
        # Compute embedding for both lists
        s1, s2 = row["questions"]["text"]

        e1 = model.encode(s1, convert_to_tensor=True)
        e2 = model.encode(s2, convert_to_tensor=True)

        cosine_scores = util.cos_sim(e1, e2)
        predictions.append(cosine_scores.item())
        actual_label.append(1 if row["is_duplicate"] else 0)

    # Calculate AUC score
    auc = roc_auc_score(actual_label, predictions)

    print(f"Model: {model_name} - AUC: {auc}")
    return auc


@stub.function(
    image=image,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secret=Secret.from_name("cohere"),
    timeout=1200,
)
async def benchmark_cohere_roc(model_name: str):
    from cohere import Client
    from cohere.responses import Embeddings
    from datasets import load_from_disk
    from sentence_transformers import util
    from sklearn.metrics import roc_auc_score
    import asyncio
    import time

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    train_test_split = dataset.train_test_split(test_size=TEST_PERCENTAGE, seed=42)
    test_dataset = train_test_split["test"].select(range(MAXIMUM_ELEMENTS_TO_TEST))
    co = Client(os.environ["COHERE_API_KEY"])

    predictions = []
    actual_label = []

    async def embed_text(row):
        response: Embeddings = co.embed(
            texts=row["questions"]["text"],
            model=model_name,
            input_type="clustering",
        )
        e1, e2 = response.embeddings
        cosine_scores = util.cos_sim(e1, e2)
        return cosine_scores.item(), 1 if row["is_duplicate"] else 0

    start = time.time()
    res = await asyncio.gather(*[embed_text(row) for row in test_dataset])
    end = time.time()
    print(f"Processed {len(test_dataset)} embeddings in {end-start}s")

    predictions = [i[0] for i in res]
    actual_label = [i[1] for i in res]

    # Calculate AUC score
    auc = roc_auc_score(actual_label, predictions)

    return auc


@stub.local_entrypoint()
def main():
    from tabulate import tabulate

    models = [
        "llmrails/ember-v1",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-large",
        "infgrad/stella-base-en-v2",
        "sentence-transformers/gtr-t5-large",
    ]

    cohere_models = [
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-light-v3.0",
    ]

    res = {}

    for model_name, auc in zip(
        models, benchmark_mteb_model.map(models, order_outputs=True)
    ):
        res[model_name] = auc

    for model_name, auc in zip(
        cohere_models, benchmark_cohere_roc.map(cohere_models, order_outputs=True)
    ):
        res[f"{model_name} (Cohere)"] = auc

    print(tabulate([[model, auc] for model, auc in res.items()], ["Model Name", "AUC"]))
