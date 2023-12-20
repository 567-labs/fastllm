from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models,
)
from torch import nn
import pathlib
from typing import Optional
import os


# TODO: refactor this to separate finetune from modal code.
# maybe initialize the dataset inside the function?
# also maybe make the hyperparameters as parameters of the finetune function
def finetune(
    save_path: pathlib.Path,
    model_id: str = "BAAI/bge-small-en-v1.5",  # Huggingface Sentence Transformers model ID
    epochs: int = 10,
    dataset_fraction: int = 1,
    activation_function=nn.Tanh(),
    scheduler: str = "warmuplinear",
    dense_out_features: Optional[int] = None,
):
    """
    Finetune a sentence transformer on the quora pairs dataset. Evaluates model performance before/after training

    :returns: evaluation accuracy post training
    :rtype: float
    Inspired by: https://github.com/UKPLab/sentence-transformers/blob/657da5fe23fe36058cbd9657aec6c7688260dd1f/examples/training/quora_duplicate_questions/training_MultipleNegativesRankingLoss.py
    """

    # TODO: handle if optuna saves to multiple checkpoints?

    # Quora pairs dataset: https://huggingface.co/datasets/quora
    DATASET_ID = "quora"
    dataset = load_dataset(DATASET_ID, split="train")
    # Quora pairs dataset only contains a "train" split in huggingface, so we will manually split it into train and test
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    if dense_out_features:
        embedding_model = SentenceTransformer(model_id)
        dense_model = models.Dense(
            in_features=embedding_model.get_sentence_embedding_dimension(),
            out_features=dense_out_features,
            activation_function=activation_function,
        )
        model = SentenceTransformer(modules=[embedding_model, dense_model])
    else:
        model = SentenceTransformer(model_id)

    train_examples = []
    # TODO: can make this more pythonic later by removing dataset_fraction
    for i in range(train_dataset.num_rows // dataset_fraction):
        text0 = train_dataset[i]["questions"]["text"][0]
        text1 = train_dataset[i]["questions"]["text"][1]
        is_duplicate = int(train_dataset[i]["is_duplicate"])
        train_examples.append(InputExample(texts=[text0, text1], label=is_duplicate))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    train_loss = losses.OnlineContrastiveLoss(model)

    test_examples = []
    # TODO: can make this more pythonic later by removing dataset_fraction
    for i in range(test_dataset.num_rows // dataset_fraction):
        text0 = test_dataset[i]["questions"]["text"][0]
        text1 = test_dataset[i]["questions"]["text"][1]
        is_duplicate = int(test_dataset[i]["is_duplicate"])
        test_examples.append(InputExample(texts=[text0, text1], label=is_duplicate))
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples,
    )

    # TODO: fix this idk

    # TODO hack, make the directory if it doesn't exist already for optuna
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # evaluator.name is used for how the file name is saved
    evaluator.csv_file = "binary_classification_evaluation_pre_train" + "_results.csv"
    pre_train_eval = evaluator(model, output_path=str(save_path))
    print("pre train eval score:", pre_train_eval)

    evaluator.csv_file = "binary_classification_evaluation" + "_results.csv"
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        output_path=str(save_path / f"{model_id.replace('/','--')}-ft"),
        checkpoint_path=str(save_path / f"checkpoints"),
        checkpoint_save_total_limit=5,
        scheduler=scheduler
    )

    evaluator.csv_file = "binary_classification_evaluation_post_train" + "_results.csv"
    post_train_eval = evaluator(model, output_path=str(save_path))

    print("post train eval score:", post_train_eval)

    return post_train_eval


# run on local with `python main.py`
if __name__ == "__main__":
    finetune(
        model_id="BAAI/bge-small-en-v1.5",
        save_path=pathlib.Path("./"),
        dataset_fraction=1000,
    )
