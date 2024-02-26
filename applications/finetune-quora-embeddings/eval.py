from distutils.log import Log
from unittest import result

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from modal import Image, gpu, Stub, Volume, Secret

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from collections import OrderedDict

# Eval Configuration
METRICS = OrderedDict(**{
    "AUC": roc_auc_score,
    "accuracy": accuracy_score,  # This is the number of correct predictions by the model ( TP + TN )/ (# of samples)
    "precision": precision_score,  # This measures the number of positive class predicitons (TP) / (TP + FP)
    "recall": recall_score,  # This measure the number of negative class predictions (TP) / ( TP + FN )
})


MODELS = {
    "BAAI/bge-base-en-v1.5": "HUGGINFACE",
    "llmrails/ember-v1": "HUGGINGFACE",
    "thenlper/gte-large": "HUGGINGFACE",
    "text-embedding-3-small": "OPENAI",
    "text-embedding-3-medium": "OPENAI",
    "text-embedding-3-large": "OPENAI",
}

def to_model_id(filename: str) -> str:
    return os.path.basename(filename).replace("-train-cossim.arrow", "").replace("-test-cossim.arrow", "")



# Stub Configuration
GPU_CONFIG = gpu.A100()

# Volume Configuration
DATASET_VOLUME = Volume.persisted("datasets")
DATASET_DIR = "/data"
METRIC = "cosine-similarity"
METRICS_DIR = f"{DATASET_DIR}/{METRIC}"

stub = Stub("finetune")

image = Image.debian_slim().pip_install("pandas", "pyarrow", "scikit-learn")


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=86400,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secrets=[Secret.from_name("huggingface-credentials"), Secret.from_name("wandb")],
)
def scorer(
    model_name: str,
    train_file: str,
    test_file: str,
    model= None,
):x
    import pandas as pd
    import pyarrow as pa
    import glob
    import os

    scoring_model = LogisticRegression() if model is None else model
    
    model_names = {
        to_model_id(file)
        for file in glob.glob(f"{COSINE_SIMILARITY_DIR}/*-cossim.arrow")
    }

    print(f"Models with generated cosine similarities: {model_ids}")

    results = {}
    for model_name in model_names:
        train_file = f"{DATASET_DIR}/cosine-similarity/{model_name}-train-cossim.arrow"
        train_df = pa.ipc.open_file(train_file).read_all().to_pandas()
        X_train = train_df["cosine_score"].values.reshape(-1, 1)

        test_file = f"{DATASET_DIR}/cosine-similarity/{model_name}-test-cossim.arrow"
        # Load training data
        test_df = pa.ipc.open_file(test_file).read_all().to_pandas()
        X_test = test_df["cosine_score"].values.reshape(-1, 1)

        y_train = train_df["is_duplicate"]

        # Prepare test data
        y_test = test_df["is_duplicate"]

        # Initialize the model
        model = ()

        # Train the model
        scoring_model.fit(X_train, y_train)
        y_pred = scoring_model.predict(X_test)

        results[model_name] = {
            metric: function(y_test, y_pred)
            for metric, function in METRICS.items()
        }

    return results


@stub.local_entrypoint()
def main():
    import json
    import pandas as pd

    for classification_model in [LogisticRegression(), DecisionTreeClassifier()]:
        model_evals_df = scorer.remote(
            model=classification_model
        )
        # results[model] = {
        #     "accuracy": accuracy_score(y_test, predictions),  
        #     "precision": precision_score(y_test, predictions),
        #     "recall": recall_score(y_test, predictions),
        #     "AUC": roc_auc_score(y_test, predictions)}

        with open(f"{model.__name__}output.json", "w") as f:
            json.dump(results, f, indent=2)
        values = []

        # just use a pandas dataframe
        for model, result, in results_dict.items():
            model_evals = [model] + [result[name] for name in METRIC.keys()]
            values.append(model_evals)
        values.sort(key=lambda x: x[1], reverse=True)
        print(tabulate(values, ["Model Name", *eval_metrics]))
