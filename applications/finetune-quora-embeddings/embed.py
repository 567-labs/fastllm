import os
from IPython import embed
from regex import D

from torch import Generator
from modal import Volume, Image, Stub, gpu, Secret
from helpers.models import EmbeddingModel, Provider
import tenacity
from datasets import Dataset

# DATASET_CONFIG
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.persisted("datasets")
CACHE_DIRECTORY = f"{DATASET_DIR}/cached-embeddings"


MODEL_TO_PROVIDER = {
    "BAAI/bge-base-en-v1.5": Provider.HUGGINGFACE,
    "BAAI/bge-small-en-v1.5": Provider.HUGGINGFACE,
    "text-embedding-3-small": Provider.OPENAI,
    "text-embedding-3-large": Provider.OPENAI,
    "text-embedding-ada-002": Provider.OPENAI,
    "embed-multilingual-v3.0": Provider.COHERE,
}

N_JOBS = 30

BATCH_SIZE_CONFIG: dict[Provider, int] = {
    Provider.HUGGINGFACE: 64,
    Provider.OPENAI: 64,
    Provider.COHERE: float("inf"),
}

GPU_CONFIG = gpu.A10G()


def download_model():
    from sentence_transformers import SentenceTransformer

    for model_name, provider in MODEL_TO_PROVIDER.items():
        if provider == Provider.HUGGINGFACE:
            print(f"Downloading and caching model: {model_name}")
            SentenceTransformer(model_name)


def has_embedding_cache(model_name):
    if MODEL_TO_PROVIDER[model_name] == Provider.HUGGINGFACE:
        model_id = model_name.split("/")[-1]
    else:
        model_id = model_name

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


def return_sentence_batchs(
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


def yield_dataset_with_embeddings(
    dataset: Dataset,
    s2id: dict[str, int],
    sentence_embeddings: dict[str, list[float]],
) -> Generator:
    for row in dataset:
        # s is the sentence
        # h is the hash
        # id is the id from the original dataset
        s1, s2 = row["questions"]["text"]
        h1, h2 = hash(s1), hash(s2)
        id1, id2 = s2id[h1], s2id[h2]

        yield {
            "id1": id1,
            "id2": id2,
            "embedding1": sentence_embeddings[id1],
            "embedding2": sentence_embeddings[id2],
            "is_duplicate": bool(row["is_duplicate"] == 1),
        }


# pd.DataFrame(
#     yield_dataset_with_embeddings(
#         train_dataset, sentence_to_id_map, sentence_embeddings
#     ).to_pyarrow()
# columns:  id1, id2, embedding1, embedding2, is_duplicate


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME})
def download_dataset():
    from datasets import load_dataset

    dataset_path = f"{DATASET_DIR}/{DATASET_NAME}"

    if os.path.exists(dataset_path):
        return

    dataset = load_dataset(DATASET_NAME)

    dataset.save_to_disk(dataset_path)
    DATASET_VOLUME.commit()


@classmethod
def from_name(cls, model_name: str) -> EmbeddingModel:
    if MODEL_TO_PROVIDER[model_name] == Provider.HUGGINGFACE:
        return EmbeddingModel.from_hf(model_name)

    if MODEL_TO_PROVIDER[model_name] == Provider.COHERE:
        return EmbeddingModel.from_cohere(model_name)

    if MODEL_TO_PROVIDER[model_name] == Provider.OPENAI:
        return EmbeddingModel.from_openai(model_name, max_limit=20)

    raise ValueError(
        f"Invalid Model of {model_name} was supplied to embed_dataset function"
    )


async def process_embeddings(
    embed_model,
    combined_dataset,
    combined_num_rows,
    sentence_to_id_map=None,
    sentence_embeddings=None,
):
    if sentence_to_id_map is None:
        sentence_to_id_map = {}

    if sentence_embeddings is None:
        sentence_embeddings = []  # should have some idea of IDS

    retrying = tenacity.Retrying(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
    )
    for attempt in retrying:
        with attempt:
            try:
                batch_size = BATCH_SIZE_CONFIG[embed_model.provider]
                sentences = return_sentence_batchs(
                    combined_dataset, sentence_to_id_map, batch_size
                )
                sentence_embeddings = await embed_model.embed(sentences)
                if len(sentence_embeddings) == combined_num_rows:
                    break
            except Exception as e:
                print(f"Error occurred while creating embeddings: {e}")
                raise e
    return sentence_to_id_map, sentence_embeddings_map


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secrets=[
        Secret.from_name("openai"),
        Secret.from_name("cohere"),
    ],
    timeout=86400,
)
async def split_embed_train_test(model_name: str):
    from datasets import load_from_disk, concatenate_datasets

    # We verify if the model has already been embedded
    if has_embedding_cache(model_name):
        print(f"Embedding has already been generated for {model_name}")
        return

    # Load the dataset for embedding
    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    embed_model = EmbeddingModel(model_name)

    combined_dataset = concatenate_datasets([dataset["test"], dataset["train"]])

    # we've precomputed the combined number of rows
    combined_rows = 408651

    sentence_to_id_map, sentence_embeddings = await process_embeddings(
        embed_model, combined_dataset, combined_num_rows=combined_rows
    )

    train_generator = yield_dataset_with_embeddings(
        dataset["train"], sentence_to_id_map, sentence_embeddings
    )

    test_generator = yield_dataset_with_embeddings(
        dataset["test"], sentence_to_id_map, sentence_embeddings
    )

    return train_generator, test_generator


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME}, timeout=2400)
def generate_embeddings(model_name):
    train_dir = f"{CACHE_DIRECTORY}/{model_name}-train.arrow"
    test_dir = f"{CACHE_DIRECTORY}/{model_name}-test.arrow"

    if has_embedding_cache(model_name):
        print(f"Embedding has already been generated for {model_name}")
        continue

    train_dataset, test_dataset = split_embed_train_test(model_name)

    start = time.time()
    for split, dataset_generator in [
        ("train", train_dataset),
        ("test", test_dataset),
    ]:
        print(f"saving {split=} for {model_name=}")
        for dataset in dataset_generator:
            print(dataset)
            pass
    total_time = time.time() - start

    try:
        DATASET_VOLUME.commit()
        print("Succesfully saved changes")
        saved = True
    except Exception as e:
        print(f"Error occurred while saving changes: {e}")
        saved = False
        raise e

    return {
        "train": train_dir,
        "test": test_dir,
        "time (s)": round(total_time, 4),
        "model": model_name,
        "is_successful": saved,
    }


def model_slug(model_name):
    if MODEL_TO_PROVIDER[model_name] == Provider.HUGGINGFACE:
        return model_name.split("/").pop()
    return model_name


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME}, timeout=2400)
def validate_dataset():
    from datasets import load_from_disk, concatenate_datasets

    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")
    test_dataset, train_dataset, val_dataset = (
        dataset["test"],
        dataset["train"],
        dataset["val"],
    )
    combined_dataset = concatenate_datasets([test_dataset, train_dataset, val_dataset])

    for idx, row in enumerate(combined_dataset):
        s1, s2 = row["questions"]["text"]

        if s1 == "" or s2 == "":
            raise ValueError(f"Found a duplicate row in row {idx}")


@stub.local_entrypoint()
def main():
    # download_dataset.remote()
    for resp in generate_embeddings.map(
        [model for model in MODEL_TO_PROVIDER.keys() if not has_embedding_cache(model)]
    ):
        print(resp)
