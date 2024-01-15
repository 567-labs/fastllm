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
def generate_dataset_split():
    from datasets import load_dataset, DatasetDict

    dataset_path = f"{DATASET_DIR}/{DATASET_NAME}"

    if os.path.exists(dataset_path):
        return

    dataset = load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, num_proc=6
    )

    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)
    train = train_test_split["train"]
    test_val_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)
    test = test_val_split["train"]
    val = test_val_split["test"]

    new_ds = DatasetDict({"train": train, "test": test, "val": val})

    new_ds.save_to_disk(dataset_path)

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
    image=image, volumes={DATASET_DIR: DATASET_VOLUME}, gpu=GPU_CONFIG, timeout=1200
)
def benchmark_mteb_model(model_name):
    from datasets import load_from_disk
    from sentence_transformers import util, SentenceTransformer
    from sklearn.metrics import roc_auc_score
    import time
    import numpy as np

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    val_dataset = dataset["val"]

    model = SentenceTransformer(model_name)
    start = time.time()

    # Process texts in batches
    texts1 = [row["questions"]["text"][0] for row in val_dataset]
    texts2 = [row["questions"]["text"][1] for row in val_dataset]
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    optimized_predictions = np.diag(cosine_scores.cpu()).tolist()
    optimized_labels = [1 if row["is_duplicate"] else 0 for row in val_dataset]

    print(f"Generated calculations in {time.time()-start}s")

    # Calculate AUC score
    return roc_auc_score(optimized_labels, optimized_predictions)


@stub.function(
    image=image,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secret=Secret.from_name("cohere"),
    timeout=1200,
)
async def benchmark_cohere_roc():
    from cohere import Client
    from datasets import load_from_disk
    from sentence_transformers import util
    from sklearn.metrics import roc_auc_score
    import asyncio
    import time

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    val_dataset = dataset["val"]
    co = Client(os.environ["COHERE_API_KEY"])
    sem = asyncio.Semaphore(32)
    predictions = []
    actual_label = []

    async def embed_text(row):
        async with sem:
            response = co.embed(
                texts=row["questions"]["text"],
                model="embed-multilingual-v3.0",
                input_type="clustering",
            )
            e1, e2 = response.embeddings
            cosine_scores = util.cos_sim(e1, e2)
            return cosine_scores.item(), 1 if row["is_duplicate"] else 0

    start = time.time()
    res = await asyncio.gather(*[embed_text(row) for row in val_dataset])
    end = time.time()
    print(f"Processed {len(val_dataset)} embeddings in {end-start}s")

    predictions = [i[0] for i in res]
    actual_label = [i[1] for i in res]

    # Calculate AUC score
    auc = roc_auc_score(actual_label, predictions)

    return auc


@stub.local_entrypoint()
def main():
    generate_dataset_split.remote()
    from tabulate import tabulate

    models = [
        "llmrails/ember-v1",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-large",
        "infgrad/stella-base-en-v2",
        "sentence-transformers/gtr-t5-large",
    ]

    res = {}
    for model_name, auc in zip(
        models, benchmark_mteb_model.map(models, order_outputs=True)
    ):
        res[model_name] = auc

    res["embed-multilingual-v3.0"] = benchmark_cohere_roc.remote()

    values = [[model, auc] for model, auc in res.items()]
    values.sort(lambda x: x[1], reverse=True)
    print(tabulate(values, ["Model Name", "AUC"]))
