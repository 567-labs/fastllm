from modal import Image, gpu, Stub, Volume, Secret


# Stub Configuration
MODEL_ID = "BAAI/bge-base-en-v1.5"
GPU_CONFIG = gpu.A100()

stub = Stub("finetune")


def download_model_and_dataset():
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    SentenceTransformer(MODEL_ID)
    load_dataset("quora", "default", "train", num_proc=6)


image = (
    Image.debian_slim()
    .pip_install("sentence-transformers", "torch", "datasets")
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


@stub.function(image=image, gpu=GPU_CONFIG)
def train_model():
    from sentence_transformers import (
        SentenceTransformer,
        InputExample,
        losses,
        evaluation,
    )
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer(MODEL_ID)
    dataset = load_dataset("quora", "default", "train")["train"]
    # Define your train examples. You need more than just two examples...
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = generate_quora_input_example(
        train_test_split["train"].select(range(5000))
    )
    test_dataset = generate_quora_input_example(train_test_split["test"])

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.OnlineContrastiveLoss(model)
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_dataset,
    )

    loss = evaluator(model)

    for i in range(10):
        # Tune the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
        )
        new_loss = evaluator(model)
        print(f"Iteration {i}: {new_loss}")

        # Just train until we get no more improvement
        if abs(new_loss - loss) < 0.01:
            break


@stub.local_entrypoint()
def main():
    train_model.remote()
