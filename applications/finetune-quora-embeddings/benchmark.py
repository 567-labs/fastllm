from modal import Image, Stub, Volume, Secret, gpu
import os

# Model Configuration
MODELS = [
    "llmrails/ember-v1",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-large",
    "infgrad/stella-base-en-v2",
    "sentence-transformers/gtr-t5-large",
]
GPU_CONFIG = gpu.A100()


# Dataset Configuration
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
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
    from datasets import load_dataset, load_from_disk

    dataset_path = f"{DATASET_DIR}/{DATASET_NAME}"

    if os.path.exists(dataset_path):
        dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
        return

    dataset = load_dataset(DATASET_NAME)

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
    image=image, volumes={DATASET_DIR: DATASET_VOLUME}, gpu=GPU_CONFIG, timeout=1200
)
def benchmark_mteb_model(model_name):
    from datasets import load_from_disk
    from sentence_transformers import util, SentenceTransformer
    from sklearn.metrics import roc_auc_score
    import time
    import numpy as np
    import torch.nn as nn
    import torch

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    test_dataset = dataset["test"]

    model = SentenceTransformer(model_name)

    sentences = []
    sentences_id_to_embedding_mapping = {}
    t1 = []
    t2 = []
    labels = []
    for row in test_dataset:
        s1, s2 = row["questions"]["text"]
        id_1, id_2 = row["questions"]["id"]

        if id_1 not in sentences_id_to_embedding_mapping:
            sentences_id_to_embedding_mapping[id_1] = len(
                sentences_id_to_embedding_mapping
            )
            sentences.append(s1)

        if id_2 not in sentences_id_to_embedding_mapping:
            sentences_id_to_embedding_mapping[id_2] = len(
                sentences_id_to_embedding_mapping
            )
            sentences.append(s2)

        t1.append(sentences_id_to_embedding_mapping[id_1])
        t2.append(sentences_id_to_embedding_mapping[id_2])
        labels.append(1 if row["is_duplicate"] else 0)

    embeddings = model.encode(sentences)
    embeddings_tensor = torch.tensor(embeddings).to("cuda")

    # Create an embedding layer with pre-trained weights
    embedding_layer = nn.Embedding.from_pretrained(embeddings_tensor)

    t1_tensor = torch.as_tensor(t1, device="cuda")
    e1 = embedding_layer(t1_tensor)

    t2_tensor = torch.as_tensor(t2, device="cuda")
    e2 = embedding_layer(t2_tensor)

    cosine_scores = util.cos_sim(e1, e2)
    predictions = np.diag(cosine_scores.cpu()).tolist()
    return roc_auc_score(labels, predictions)


@stub.function(
    image=image,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secret=Secret.from_name("openai"),
    timeout=86400,
    gpu=GPU_CONFIG,
)
async def benchmark_openai():
    from datasets import load_from_disk
    from sentence_transformers import util
    from sklearn.metrics import roc_auc_score
    import asyncio
    import time
    from tqdm import tqdm
    from openai import AsyncOpenAI
    import numpy as np
    import torch
    import torch.nn as nn

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    test_dataset = dataset["test"]

    client = AsyncOpenAI()

    sem = asyncio.Semaphore(20)
    sentences = []
    sentences_id_to_embedding_mapping = {}
    t1 = []
    t2 = []
    labels = []
    for row in test_dataset:
        s1, s2 = row["questions"]["text"]
        id_1, id_2 = row["questions"]["id"]

        if id_1 not in sentences_id_to_embedding_mapping:
            sentences_id_to_embedding_mapping[id_1] = len(
                sentences_id_to_embedding_mapping
            )
            sentences.append(s1)

        if id_2 not in sentences_id_to_embedding_mapping:
            sentences_id_to_embedding_mapping[id_2] = len(
                sentences_id_to_embedding_mapping
            )
            sentences.append(s2)

        t1.append(sentences_id_to_embedding_mapping[id_1])
        t2.append(sentences_id_to_embedding_mapping[id_2])
        labels.append(1 if row["is_duplicate"] else 0)

    batch_size = 16
    print(f"Extracted {len(sentences)} unique sentences")
    tqdm_monitoring_bar = tqdm(total=len(sentences))

    async def embed_text(texts, progress_bar: tqdm):
        async with sem:
            response = await client.embeddings.create(
                input=texts, model="text-embedding-ada-002"
            )
            progress_bar.update(len(texts))
            return [item.embedding for item in response.data]

    start = time.time()
    coros = [
        embed_text(sentences[start : start + batch_size], tqdm_monitoring_bar)
        for start in range(0, len(sentences), batch_size)
    ]
    res = await asyncio.gather(*coros)
    tqdm_monitoring_bar.close()
    end = time.time()

    flattened_res = [item for sublist in res for item in sublist]
    print(f"Processed {len(flattened_res)} embeddings in {end-start}s")
    embeddings_tensor = torch.tensor(flattened_res).to("cuda")
    embedding_layer = nn.Embedding.from_pretrained(embeddings_tensor)

    t1_tensor = torch.as_tensor(t1, device="cuda")
    e1 = embedding_layer(t1_tensor)

    t2_tensor = torch.as_tensor(t2, device="cuda")
    e2 = embedding_layer(t2_tensor)

    cosine_scores = util.cos_sim(e1, e2)
    predictions = np.diag(cosine_scores.cpu()).tolist()
    return roc_auc_score(labels, predictions)


@stub.function(
    image=image,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secret=Secret.from_name("cohere"),
    timeout=86400,
    gpu=GPU_CONFIG,
)
async def benchmark_cohere_roc():
    from cohere import Client
    from datasets import load_from_disk
    from sentence_transformers import util
    from sklearn.metrics import roc_auc_score
    import asyncio
    import time
    from tqdm import tqdm
    import numpy as np
    import torch
    import torch.nn as nn

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    test_dataset = dataset["test"]
    co = Client(os.environ["COHERE_API_KEY"])
    sem = asyncio.Semaphore(64)
    sentences = []
    sentences_id_to_embedding_mapping = {}
    t1 = []
    t2 = []
    labels = []
    for row in test_dataset:
        s1, s2 = row["questions"]["text"]
        id_1, id_2 = row["questions"]["id"]

        if id_1 not in sentences_id_to_embedding_mapping:
            sentences_id_to_embedding_mapping[id_1] = len(
                sentences_id_to_embedding_mapping
            )
            sentences.append(s1)

        if id_2 not in sentences_id_to_embedding_mapping:
            sentences_id_to_embedding_mapping[id_2] = len(
                sentences_id_to_embedding_mapping
            )
            sentences.append(s2)

        t1.append(sentences_id_to_embedding_mapping[id_1])
        t2.append(sentences_id_to_embedding_mapping[id_2])
        labels.append(1 if row["is_duplicate"] else 0)

    batch_size = 96
    print(f"Extracted {len(sentences)} unique sentences")
    tqdm_monitoring_bar = tqdm(total=len(sentences))

    async def embed_text(texts, progress_bar: tqdm):
        async with sem:
            response = co.embed(
                texts=texts,
                model="embed-multilingual-v3.0",
                input_type="clustering",
            )
            progress_bar.update(len(texts))
            return response.embeddings

    start = time.time()
    coros = [
        embed_text(sentences[start : start + batch_size], tqdm_monitoring_bar)
        for start in range(0, len(sentences), batch_size)
    ]
    res = await asyncio.gather(*coros)
    tqdm_monitoring_bar.close()
    end = time.time()
    print(f"Generated embeddings in {end-start}s")

    # Flatten the list of lists and convert to tensor of floats
    flattened_res = [item for sublist in res for item in sublist]
    embeddings_tensor = torch.tensor(flattened_res).to("cuda")
    embedding_layer = nn.Embedding.from_pretrained(embeddings_tensor)

    t1_tensor = torch.as_tensor(t1, device="cuda")
    e1 = embedding_layer(t1_tensor)

    t2_tensor = torch.as_tensor(t2, device="cuda")
    e2 = embedding_layer(t2_tensor)

    cosine_scores = util.cos_sim(e1, e2)
    predictions = np.diag(cosine_scores.cpu()).tolist()
    return roc_auc_score(labels, predictions)


@stub.local_entrypoint()
def main():
    from tabulate import tabulate

    generate_dataset_split.remote()

    res = {}
    res["text-embeddings-ada-v2"] = benchmark_openai.remote()
    res["embed-multilingual-v3.0"] = benchmark_cohere_roc.remote()

    for model_name, auc in zip(
        MODELS, benchmark_mteb_model.map(MODELS, order_outputs=True)
    ):
        res[model_name] = auc

    values = [[model, auc] for model, auc in res.items()]
    values.sort(key=lambda x: x[1], reverse=True)
    print(tabulate(values, ["Model Name", "AUC"]))
