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
    "cohere", "datasets", "sentence-transformers", "scikit-learn", "tabulate", "openai"
)


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME})
def generate_dataset_split():
    from datasets import load_dataset, DatasetDict, load_from_disk

    dataset_path = f"{DATASET_DIR}/{DATASET_NAME}"

    if os.path.exists(dataset_path):
        dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
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
    secret=Secret.from_name("openai"),
    timeout=86400,
)
async def benchmark_openai():
    from datasets import load_from_disk
    from sentence_transformers import util
    from sklearn.metrics import roc_auc_score
    import asyncio
    import time
    from tqdm import tqdm
    from openai import OpenAI

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    val_dataset = dataset["val"]

    client = OpenAI()

    sem = asyncio.Semaphore(64)
    predictions = []
    actual_label = []

    tqdm_monitoring_bar = tqdm(total=len(val_dataset))

    async def embed_text(row, progress_bar: tqdm):
        async with sem:
            response = client.embeddings.create(
                input=row["questions"]["text"], model="text-embedding-ada-002"
            ).data
            e1 = response[0].embedding
            e2 = response[1].embedding
            cosine_scores = util.cos_sim(e1, e2)
            progress_bar.update(1)
            return cosine_scores.item(), 1 if row["is_duplicate"] else 0

    start = time.time()
    res = await asyncio.gather(
        *[embed_text(row, tqdm_monitoring_bar) for row in val_dataset]
    )
    tqdm_monitoring_bar.close()
    end = time.time()
    print(f"Processed {len(val_dataset)} embeddings in {end-start}s")

    predictions = [i[0] for i in res]
    actual_label = [i[1] for i in res]

    # Calculate AUC score
    auc = roc_auc_score(actual_label, predictions)

    return auc


@stub.function(
    image=image,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secret=Secret.from_name("cohere"),
    timeout=86400,
)
async def benchmark_cohere_roc():
    from cohere import Client
    from datasets import load_from_disk
    from sentence_transformers import util
    from sklearn.metrics import roc_auc_score
    import asyncio
    import time
    from tqdm import tqdm

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    val_dataset = dataset["val"]
    co = Client(os.environ["COHERE_API_KEY"])
    sem = asyncio.Semaphore(64)
    predictions = []
    actual_label = []

    tqdm_monitoring_bar = tqdm(total=len(val_dataset))

    async def embed_text(row, progress_bar: tqdm):
        async with sem:
            response = co.embed(
                texts=row["questions"]["text"],
                model="embed-multilingual-v3.0",
                input_type="clustering",
            )
            e1, e2 = response.embeddings
            cosine_scores = util.cos_sim(e1, e2)
            progress_bar.update(1)
            return cosine_scores.item(), 1 if row["is_duplicate"] else 0

    start = time.time()
    res = await asyncio.gather(
        *[embed_text(row, tqdm_monitoring_bar) for row in val_dataset]
    )
    tqdm_monitoring_bar.close()
    end = time.time()
    print(f"Processed {len(val_dataset)} embeddings in {end-start}s")

    predictions = [i[0] for i in res]
    actual_label = [i[1] for i in res]

    # Calculate AUC score
    auc = roc_auc_score(actual_label, predictions)

    return auc


@stub.local_entrypoint()
def main():
    from tabulate import tabulate

    generate_dataset_split.remote()

    res = {}
    res["text-embeddings-ada-v2"] = benchmark_openai.remote()
    # res["embed-multilingual-v3.0"] = benchmark_cohere_roc.remote()
    # models = [
    #     "llmrails/ember-v1",
    #     "BAAI/bge-base-en-v1.5",
    #     "thenlper/gte-large",
    #     "infgrad/stella-base-en-v2",
    #     "sentence-transformers/gtr-t5-large",
    #     "567-labs/bge-base-en-v1.5-ft-quora-0.9",
    #     "567-labs/bge-base-en-v1.5-ft-quora-0.7",
    #     "567-labs/bge-base-en-v1.5-ft-quora-0.5",
    #     "567-labs/bge-base-en-v1.5-ft-quora-0.3",
    #     "567-labs/bge-base-en-v1.5-ft-quora-0.1",
    # ]

    # for model_name, auc in zip(
    #     models, benchmark_mteb_model.map(models, order_outputs=True)
    # ):
    #     res[model_name] = auc

    values = [[model, auc] for model, auc in res.items()]
    values.sort(key=lambda x: x[1], reverse=True)
    print(tabulate(values, ["Model Name", "AUC"]))
