from sentence_transformers import SentenceTransformer
import pandas as pd


def format_dataset(dataset):
    from sentence_transformers import InputExample

    return [
        InputExample(
            texts=row["questions"]["text"], label=1 if row["is_duplicate"] else 0
        )
        for row in dataset
    ]


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


def flatten_data(dataset):
    seen = set()
    for pairs in dataset["questions"]:
        for id, text in zip(pairs["id"], pairs["text"]):
            if id not in seen:
                seen.add(id)
                yield {"id": id, "text": text}


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


def score_prediction(model, train_dataset, test_dataset):
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
