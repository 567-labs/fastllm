from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from modal import gpu, Volume, Image, Stub, Secret
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
from torch.utils.data import DataLoader

# MODAL CONFIG
GPU_CONFIG = gpu.A100()

# DATASET CONFIG
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
DATASET_DIR = "/data"
DATASET_VOLUME = Volume.persisted("datasets")
CACHE_DIRECTORY = f"{DATASET_DIR}/cached-embeddings"

# MODEL CONFIG
MODEL_ID = "BAAI/bge-base-en-v1.5"
MODEL_OUT_LAYER_DIM = 256

# Training Configuration
BATCH_SIZE = 64

# Eval Configuration
METRICS = {
    "accuracy": accuracy_score,  # This is the number of correct predictions by the model ( TP + TN )/ (# of samples)
    "precision": precision_score,  # This measures the number of positive class predicitons (TP) / (TP + FP)
    "recall": recall_score,  # This measure the number of negative class predictions (TP) / ( TP + FN )
    "AUC": roc_auc_score,
}


def download_model():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_ID)


image = (
    Image.debian_slim()
    .pip_install("datasets", "sentence-transformers", "scikit-learn", "pandas", "torch")
    .run_function(download_model)
)

stub = Stub("finetune")


def flatten_data(dataset):
    seen = set()
    for pairs in dataset["questions"]:
        for id, text in zip(pairs["id"], pairs["text"]):
            if id not in seen:
                seen.add(id)
                yield {"id": id, "text": text}


def embed(model: SentenceTransformer, text_to_embed) -> pd.DataFrame:
    """
    This is a generic function which takes in a sentence transformer model and a generator which generates results in the form of (id,text). It then returns a pandas dataframe which contains the relevant embeddings

    Parameters
    - model ( SentenceTransformer ): This is a sentence transformer model
    - text_to_embed: This is a generator which contains python objects that have the keys id and text
    """
    import numpy as np

    texts = []
    ids = []
    for row in text_to_embed:
        texts.append(row["text"])
        ids.append(row["id"])

    results = [
        (id, embedding) for embedding, id in zip(model.encode(np.array(texts)), ids)
    ]
    return pd.DataFrame(results, columns=["id", "embedding"]).set_index("id")


def new_dataset(embeddings: pd.DataFrame, dataset):
    """
    This is a dataset that takes in a existing dataset and then generates a new pandas dataset for use down the line.
    """
    for data in dataset:
        try:
            id1, id2 = data["questions"]["id"]
            e1, e2 = embeddings.loc[id1].iloc[0], embeddings.loc[id2].iloc[0]
            yield (id1, id2, e1, e2, data["is_duplicate"])
        except Exception as e:
            print(data)
            embedding_ids = embeddings.index.tolist()
            # Check if id1 and id2 are in the list of embedding ids
            id1_exists = id1 in embedding_ids
            id2_exists = id2 in embedding_ids
            print(f"id1 exists: {id1_exists}, id2 exists: {id2_exists}")
            print(f"Encountered an exception: {e}")
            raise e


def generate_cosine_similarity(dataset):
    """
    This function assumes that you have a dataset that contains at least these 3 columns

    - is_duplicate : Which indicates the label that determines if the two embeddings are similar or not
    - e1 : Which is the embedding of the first sentence
    - e2 : which is the embedding of the second sentence
    """
    from tqdm import tqdm
    import numpy as np
    from sentence_transformers import util
    import torch

    assert set(["e1", "e2", "is_duplicate"]).issubset(
        dataset.columns
    ), "Dataset does not have the required columns"

    embeddings_first_sentence = dataset["e1"]
    embeddings_second_sentence = dataset["e2"]
    labels = dataset["is_duplicate"]

    batch_size = 2000
    num_batches = int(np.ceil(len(embeddings_first_sentence) / batch_size))
    predictions = []
    for idx in tqdm(range(num_batches)):
        start_index = idx * batch_size
        end_index = min((idx + 1) * batch_size, len(embeddings_first_sentence))

        batch_embeddings_sentence_1 = np.stack(
            embeddings_first_sentence[start_index:end_index].values
        )
        batch_embeddings_sentence_2 = np.stack(
            embeddings_second_sentence[start_index:end_index].values
        )

        cosine_scores = util.cos_sim(
            torch.Tensor(batch_embeddings_sentence_1),
            torch.Tensor(batch_embeddings_sentence_2),
        )
        batch_predictions = np.diag(cosine_scores.cpu()).tolist()
        predictions.extend(batch_predictions)

    return predictions, labels


def format_dataset(dataset):
    from sentence_transformers import InputExample

    return [
        InputExample(
            texts=row["questions"]["text"], label=1 if row["is_duplicate"] else 0
        )
        for row in dataset
    ]


def generate_prediction_labels(model, train_dataset, test_dataset):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    from datasets import concatenate_datasets

    concatenated_dataset = concatenate_datasets([train_dataset, test_dataset])
    embeddings = embed(model, flatten_data(concatenated_dataset))
    test_dataset_with_embeddings = pd.DataFrame(
        new_dataset(embeddings, test_dataset),
        columns=["id1", "id2", "e1", "e2", "is_duplicate"],
    )
    train_dataset_with_embeddings = pd.DataFrame(
        new_dataset(embeddings, train_dataset),
        columns=["id1", "id2", "e1", "e2", "is_duplicate"],
    )

    # Now we calculate the cosine similarities and the labels
    train_preds, train_labels = generate_cosine_similarity(
        train_dataset_with_embeddings
    )
    test_preds, test_labels = generate_cosine_similarity(test_dataset_with_embeddings)

    # # We then calculate the loss by fitting a custom LR
    lr = LogisticRegression()
    lr.fit(np.array(train_preds).reshape(-1, 1), train_labels)
    return lr.predict(np.array(test_preds).reshape(-1, 1)), test_labels


@stub.function(
    image=image, gpu=GPU_CONFIG, volumes={DATASET_DIR: DATASET_VOLUME}, timeout=86400
)
def finetune_model(dataset_pct: float):
    from datasets import load_from_disk
    from sentence_transformers import SentenceTransformer, losses, evaluation, models
    import math
    from torch import nn

    assert (
        isinstance(dataset_pct, float) and 0 < dataset_pct <= 1
    ), f"Invalid value of {dataset_pct} provided."

    # Load in the dataset
    dataset = load_from_disk(f"{DATASET_DIR}/{DATASET_NAME}")

    print(f"Dataset Train Size: {len(dataset['train'])}")
    train_amount = math.floor(len(dataset["train"]) * dataset_pct)
    train_dataset = dataset["train"].select(range(train_amount))
    test_dataset = dataset["test"]

    print(f"Initialising Model {MODEL_ID}")

    # Initialise the model with its dense layer and the loss/evaluator objects
    model = SentenceTransformer(MODEL_ID)
    dense_layer = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=MODEL_OUT_LAYER_DIM,
        activation_function=nn.Tanh(),
    )
    model = SentenceTransformer(modules=[model, dense_layer])
    train_loss = losses.OnlineContrastiveLoss(model)
    train_examples = format_dataset(train_dataset)
    test_examples = format_dataset(test_dataset)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    print(f"Generated {len(train_examples)} train examples and {len(test_examples)}")

    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples
    )

    best_eval = (evaluator(model, output_path=None), 0)
    print(f"Initial Eval: {best_eval}")
    for iteration in range(6):
        # Train Model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=1,
        )
        curr_eval = evaluator(model, output_path=None)
        best_eval_perf, time_step = best_eval

        if curr_eval >= best_eval_perf and iteration - time_step >= 2:
            print(
                f"No Improvement detected in training loss. Final loss was {curr_eval}"
            )
            break
        print(
            f"Iteration {iteration+1}: {curr_eval}, eval: ( {best_eval_perf}, {time_step} )"
        )
        if curr_eval > best_eval_perf:
            best_eval = (curr_eval, iteration)

    predictions, test_labels = generate_prediction_labels(
        model, train_dataset, test_dataset
    )

    return dataset_pct, {
        metric: function(test_labels, predictions)
        for metric, function in METRICS.items()
    }


@stub.local_entrypoint()
def main():
    import json

    PERCENTAGES = [0.01]
    DATASET_SIZE = 261317
    cache_file = "model_perf_dense.json"
    try:
        with open(cache_file, "r") as json_file:
            model_perf = json.load(json_file)
    except FileNotFoundError:
        model_perf = {}

    for item in finetune_model.map(PERCENTAGES):
        dataset_pct, results = item
        model_perf[str(dataset_pct)] = results

    with open(cache_file, "w") as json_file:
        json.dump(model_perf, json_file, indent=4)
