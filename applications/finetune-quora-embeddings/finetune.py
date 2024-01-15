from modal import Image, gpu, Stub, Volume, Secret


# Finetune Configuration
MAXIMUM_TRAIN_DATASET_USAGE = 0.1

# Stub Configuration
MODEL_ID = "BAAI/bge-base-en-v1.5"
GPU_CONFIG = gpu.A10G()

stub = Stub("finetune")


def download_model_and_dataset():
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    SentenceTransformer(MODEL_ID)
    load_dataset("quora", "default", "train", num_proc=6)


image = (
    Image.debian_slim()
    .pip_install("sentence-transformers", "torch", "datasets", "scikit-learn", "tqdm")
    .run_function(download_model_and_dataset)
)


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


@stub.function(image=image, gpu=GPU_CONFIG, timeout=3600)
def train_model(train_dataset_percentage: float):
    from sentence_transformers import SentenceTransformer, losses, evaluation, util
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    import math
    import time
    from tqdm import tqdm
    import numpy as np

    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer(MODEL_ID)
    dataset = load_dataset("quora", "default", "train")["train"]
    # Define your train examples. You need more than just two examples...
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_slice = math.floor(
        len(train_test_split["train"]) * train_dataset_percentage
    )
    train_dataset = generate_quora_input_example(
        train_test_split["train"].select(range(dataset_slice))
    )

    test_dataset = generate_quora_input_example(train_test_split["test"])

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.OnlineContrastiveLoss(model)
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_dataset,
    )

    loss = evaluator(model)

    # for i in range(2):
    #     # Tune the model
    #     model.fit(
    #         train_objectives=[(train_dataloader, train_loss)],
    #         epochs=1,
    #         warmup_steps=100,
    #     )
    #     new_loss = evaluator(model)
    #     print(f"Iteration {i}: {new_loss}")

    #     if abs(new_loss - loss) < 0.01:
    #         break

    predictions = []
    actual_label = []
    start = time.time()
    for idx, row in enumerate(test_dataset):
        if idx % 500 == 0:
            print(f"Iteration {idx}/ {len(test_dataset)}")
        # Compute embedding for both lists
        s1, s2 = row.texts

        e1 = model.encode(s1, convert_to_tensor=True)
        e2 = model.encode(s2, convert_to_tensor=True)

        cosine_scores = util.cos_sim(e1, e2)
        predictions.append(round(cosine_scores.item(), 5))
        actual_label.append(1 if row.label else 0)

    print(f"Generated calculations in {time.time()-start}s")

    start = time.time()

    # Process texts in batches
    texts1 = [row.texts[0] for row in test_dataset]
    texts2 = [row.texts[1] for row in test_dataset]
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    optimized_predictions = np.diag(cosine_scores.cpu()).tolist()
    optimized_labels = [1 if row.label else 0 for row in test_dataset]

    print(f"Generated calculations in {time.time()-start}s")

    # Calculate AUC score
    # 1000 - 0.8617349327217518, 0.8617327853151748 (diff 0.000002)
    # 10,000
    auc = roc_auc_score(actual_label, predictions)
    auc_prime = roc_auc_score(optimized_labels, optimized_predictions)

    print(auc, auc_prime)


@stub.local_entrypoint()
def main():
    percentages = [
        0.1,
    ]
    res = {}
    for percentage, auc in zip(
        percentages, train_model.map(percentages, order_outputs=True)
    ):
        res[percentage] = auc

    for percentage in percentages:
        print(res[percentage])
