def batch_sentences(test_dataset, sentence_to_id_mapping: dict, batch_size=1000):
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
    dataset,
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
