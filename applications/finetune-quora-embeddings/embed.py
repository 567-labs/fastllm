from modal import Volume, Image, Stub, gpu, Secret
import os
from helpers.models import EmbeddingModel, model_types
from datasets import Dataset

# DATASET_CONFIG
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.persisted("datasets")
CACHE_DIRECTORY = f"{DATASET_DIR}/cached-embeddings"

# Model Configs
MODELS = [
    "BAAI/bge-base-en-v1.5",
    "llmrails/ember-v1",
    "thenlper/gte-large",
    "infgrad/stella-base-en-v2",
    "sentence-transformers/gtr-t5-large",
]
OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-ada-002"]

COHERE_MODELS = ["embed-multilingual-v3.0"]

SEMAPHORE_LIMITS: dict[model_types, int] = {
    "HuggingFace": 20,
    "OpenAI": 20,
    "Cohere": 20,
}

BATCH_SIZE_CONFIG: dict[model_types, int] = {
    "HuggingFace": 10000,
    "OpenAI": 64,
    "Cohere": float("inf"),
}

GPU_CONFIG = gpu.A100()


def download_model():
    from sentence_transformers import SentenceTransformer

    for model_name in MODELS:
        print(f"Downloading and caching model: {model_name}")
        SentenceTransformer(model_name)


def determine_if_model_has_generated_embeddings(model_name):
    model_id = model_name.split("/").pop() if model_name in MODELS else model_name
    train_file_path = f"{CACHE_DIRECTORY}/{model_id}-train.arrow"
    test_file_path = f"{CACHE_DIRECTORY}/{model_id}-test.arrow"
    return os.path.exists(train_file_path) and os.path.exists(test_file_path)


image = (
    Image.debian_slim()
    .env({"Random": "123"})
    .pip_install(
        "cohere",
        "datasets",
        "sentence-transformers",
        "scikit-learn",
        "tabulate",
        "openai",
        "pyarrow",
        "tenacity",
        "diskcache",
    )
    .run_function(download_model)
)

stub = Stub("embeddings")


def get_unique_sentences(
    test_dataset: Dataset, sentence_to_id_mapping: dict, batch_size=1000
):
    seen = set()
    batch = []
    for row in test_dataset:
        s1, s2 = row["questions"]["text"]

        if s1 not in seen:
            sentence_to_id_mapping[s1] = len(seen)
            seen.add(s1)
            batch.append(s1)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if s2 not in seen:
            sentence_to_id_mapping[s2] = len(seen)
            seen.add(s2)
            batch.append(s2)

            if len(batch) == batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def update_dataset_with_embeddings(
    dataset: Dataset,
    sentence_to_id_map: dict[str, int],
    sentence_embeddings,
):
    import pyarrow as pa

    # We generate a new
    dataset_questions_with_embeddings = []
    dataset_labels = []
    for row in dataset:
        s1, s2 = row["questions"]["text"]
        sentence_1_embedding_id = sentence_to_id_map[s1]
        sentence_2_embedding_id = sentence_to_id_map[s2]

        sentence_1_embedding = sentence_embeddings[sentence_1_embedding_id]
        sentence_2_embedding = sentence_embeddings[sentence_2_embedding_id]

        new_dataset_row_with_embeddings = {
            "id": row["questions"]["id"],
            "text": row["questions"]["text"],
            "embeddings": [sentence_1_embedding, sentence_2_embedding],
        }
        dataset_questions_with_embeddings.append(new_dataset_row_with_embeddings)
        dataset_labels.append(row["is_duplicate"])

    # Convert the sentences and their embeddings to a table
    return pa.Table.from_arrays(
        [
            pa.array(dataset_questions_with_embeddings),
            pa.array(dataset_labels),
        ],
        names=[
            "questions",
            "is_duplicate",
        ],
    )


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME})
def download_dataset():
    from datasets import load_dataset

    dataset_path = f"{DATASET_DIR}/{DATASET_NAME}"

    if os.path.exists(dataset_path):
        return

    dataset = load_dataset(DATASET_NAME)

    dataset.save_to_disk(dataset_path)
    DATASET_VOLUME.commit()


@stub.function(
    image=image,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secrets=[
        Secret.from_name("openai"),
        Secret.from_name("cohere"),
    ],
    timeout=86400,
)
async def embed_dataset(model_name: str):
    from datasets import load_from_disk, concatenate_datasets
    import math
    import time

    # We verify if the model has already been embedded
    if determine_if_model_has_generated_embeddings(model_name):
        print(f"Embedding has already been generated for {model_name}")
        return

    combined_num_rows = 408651  # Extract
    # First Load the model
    if model_name in MODELS:
        model: model_types = "HuggingFace"
        embed_model = EmbeddingModel.from_hf(
            model_name, math.ceil(combined_num_rows / BATCH_SIZE_CONFIG["HuggingFace"])
        )
    elif model_name in COHERE_MODELS:
        embed_model = EmbeddingModel.from_cohere(model_name)
        model = "Cohere"
    elif model_name in OPENAI_MODELS:
        model: model_types = "OpenAI"
        embed_model = EmbeddingModel.from_openai(model_name, SEMAPHORE_LIMITS["OpenAI"])
    else:
        raise ValueError(
            f"Invalid Model of {model_name} was supplied to embed_dataset function"
        )

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]
    combined_dataset = concatenate_datasets([test_dataset, train_dataset])

    sentence_to_id_map = dict()

    for idx in range(3):
        print(f"Iteration {idx}")
        try:
            sentences = get_unique_sentences(
                combined_dataset, sentence_to_id_map, BATCH_SIZE_CONFIG[model]
            )
            sentence_embeddings = await embed_model.embed(sentences)
            if len(sentence_embeddings) == combined_num_rows:
                break
        except Exception as e:
            print(
                f"Encountered Exception of {e}. Retrying embedding eneration in 30 seconds again"
            )
            time.sleep(30)

    return update_dataset_with_embeddings(
        train_dataset,
        sentence_to_id_map,
        sentence_embeddings,
    ), update_dataset_with_embeddings(
        test_dataset,
        sentence_to_id_map,
        sentence_embeddings,
    )


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME}, timeout=2400)
def generate_embeddings():
    import pyarrow as pa
    import os

    test = OPENAI_MODELS[1:]

    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)

    for model_name, res in zip(test, embed_dataset.map(test, order_outputs=True)):
        if determine_if_model_has_generated_embeddings(model_name):
            print(
                f"Cache Data for {model_name} already exists. Skipping the generation"
            )
            continue
        print(f"Generated embeddings for {model_name}")

        train_dataset, test_dataset = res

        model_id = model_name.split("/").pop() if model_name in MODELS else model_name
        with pa.OSFile(f"{CACHE_DIRECTORY}/{model_id}-train.arrow", "wb") as sink:
            writer = pa.RecordBatchFileWriter(sink, train_dataset.schema)
            writer.write_table(train_dataset)
            writer.close()

        with pa.OSFile(f"{CACHE_DIRECTORY}/{model_id}-test.arrow", "wb") as sink:
            writer = pa.RecordBatchFileWriter(sink, test_dataset.schema)
            writer.write_table(test_dataset)
            writer.close()

        print(f"Cache files generated for {model_name}")
        DATASET_VOLUME.commit()
    print("Succesfully saved changes")


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME}, timeout=2400)
def validate_dataset():
    from datasets import load_from_disk, concatenate_datasets

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    test_dataset = dataset["test"]
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    combined_dataset = concatenate_datasets([test_dataset, train_dataset, val_dataset])

    for idx, row in enumerate(combined_dataset):
        s1, s2 = row["questions"]["text"]

        if s1 == "" or s2 == "":
            raise ValueError(f"Found a duplicate row in row {idx}")


@stub.local_entrypoint()
def main():
    # download_dataset.remote()
    generate_embeddings.remote()
