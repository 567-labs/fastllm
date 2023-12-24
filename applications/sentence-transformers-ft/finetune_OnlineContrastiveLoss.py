from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models,
    LoggingHandler,
)
from torch import nn
import pathlib
from typing import Optional
import os
import logging


def finetune(
    save_path: pathlib.Path,
    model_id: str = "BAAI/bge-small-en-v1.5",
    epochs: int = 10,
    dataset_fraction: int = 1,
    activation_function=nn.Tanh(),
    scheduler: str = "warmuplinear",
    dense_out_features: Optional[int] = None,
    batch_size: int = 32,
):
    """
    Finetune a sentence transformer on the quora pairs dataset. Evaluates model performance before/after training

    :returns: evaluation accuracy post training
    :rtype: float
    Inspired by: https://github.com/UKPLab/sentence-transformers/blob/657da5fe23fe36058cbd9657aec6c7688260dd1f/examples/training/quora_duplicate_questions/training_MultipleNegativesRankingLoss.py
    """
    #### Just logging initialization
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    logger = logging.getLogger(__name__)

    #### Initialize Model
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

    #### Initialize Dataset
    # Quora pairs dataset: https://huggingface.co/datasets/quora
    DATASET_ID = "quora"
    dataset = load_dataset(DATASET_ID, split="train")
    # Quora pairs dataset only contains a "train" split in huggingface, so we will manually split it into train and test
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    # For simplicity we only split into 2 datasets, but you can add another split for "eval", 3 splits in total, if desired
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    #### Format Dataset for loss and eval
    train_examples = [
        InputExample(
            texts=[
                train_dataset[i]["questions"]["text"][0],
                train_dataset[i]["questions"]["text"][1],
            ],
            label=int(train_dataset[i]["is_duplicate"]),
        )
        for i in range(train_dataset.num_rows // dataset_fraction)
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    test_examples = [
        InputExample(
            texts=[
                test_dataset[i]["questions"]["text"][0],
                test_dataset[i]["questions"]["text"][1],
            ],
            label=int(test_dataset[i]["is_duplicate"]),
        )
        for i in range(test_dataset.num_rows // dataset_fraction)
    ]

    #### Initialize loss
    train_loss = losses.OnlineContrastiveLoss(model)

    #### Initialize evaluator
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        test_examples,
    )

    #### Pre Train Evaluation
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    evaluator.csv_file = "binary_classification_evaluation_pre_train" + "_results.csv"
    logger.info("### Model Evaluation Without Training ###")
    evaluator(model, output_path=str(save_path))
    pre_train_eval = evaluator(model, output_path=str(save_path))
    logger.info(
        f"Post train eval score (highest Average Precision across all distance functions):{pre_train_eval}"
    )

    #### Train the model
    evaluator.csv_file = "binary_classification_evaluation" + "_results.csv"
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        output_path=str(save_path / f"{model_id.replace('/','--')}-ft"),
        checkpoint_path=str(save_path / f"checkpoints"),
        checkpoint_save_total_limit=3,
        scheduler=scheduler,
        callback=lambda score, epoch, steps: logger.info(
            f"Epoch {epoch}: score {score}"
        ),
    )

    #### Evaluate the model afterwards
    # For simplicity, we are only returning metrics for the fully trained model rather than early-stopping and choosing the best performing model
    evaluator.csv_file = "binary_classification_evaluation_post_train" + "_results.csv"
    post_train_eval = evaluator(model, output_path=str(save_path))
    logger.info(
        f"Post train eval score (highest Average Precision across all distance functions):{post_train_eval}"
    )

    return post_train_eval


# run on local with `python finetune_OnlineContrastiveLoss.py`
if __name__ == "__main__":
    # hyperparameters set low as an example to run quicker on local device
    finetune(
        model_id="BAAI/bge-small-en-v1.5",
        save_path=pathlib.Path("./output"),
        dataset_fraction=100,
        epochs=5,
        batch_size=8,
    )
