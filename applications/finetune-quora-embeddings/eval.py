from modal import Image, gpu, Stub, Volume, Secret

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Eval Configuration
METRICS = {
    "accuracy": accuracy_score,  # This is the number of correct predictions by the model ( TP + TN )/ (# of samples)
    "precision": precision_score,  # This measures the number of positive class predicitons (TP) / (TP + FP)
    "recall": recall_score,  # This measure the number of negative class predictions (TP) / ( TP + FN )
    "AUC": roc_auc_score,
}

# Dataset Configuration
MODELS = ["BAAI/bge-base-en-v1.5", "llmrails/ember-v1", "thenlper/gte-large"]
OPENAI_MODELS = ["text-embedding-3-small"]

# Stub Configuration
GPU_CONFIG = gpu.A100()

# Volume Configuration
DATASET_VOLUME = Volume.persisted("datasets")
DATASET_DIR = "/data"
COSINE_SIMILARITY_DIR = f"{DATASET_DIR}/cosine-similarity"

stub = Stub("finetune")

image = Image.debian_slim().pip_install("pandas", "pyarrow", "scikit-learn")


@stub.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=86400,
    volumes={DATASET_DIR: DATASET_VOLUME},
    secrets=[Secret.from_name("huggingface-credentials"), Secret.from_name("wandb")],
)
def train_logistic_regression():
    import pandas as pd
    import pyarrow as pa
    import glob
    from sklearn.linear_model import LogisticRegression
    import os

    # Get all .arrow files in the /cosine-similarity directory
    arrow_files = glob.glob(f"{COSINE_SIMILARITY_DIR}/*-cossim.arrow")

    # Extract model ids from file names
    model_ids = set()
    for file in arrow_files:
        model_id = (
            os.path.basename(file)
            .replace("-train-cossim.arrow", "")
            .replace("-test-cossim.arrow", "")
        )
        model_ids.add(model_id)

    print(f"Models with generated cosine similarities: {model_ids}")

    results = {}
    for model_name in model_ids:
        train_file = f"{DATASET_DIR}/cosine-similarity/{model_name}-train-cossim.arrow"
        test_file = f"{DATASET_DIR}/cosine-similarity/{model_name}-test-cossim.arrow"
        # Load training data
        train_df = pa.ipc.open_file(train_file).read_all().to_pandas()
        # Load test data
        test_df = pa.ipc.open_file(test_file).read_all().to_pandas()

        # Prepare training data
        X_train = train_df["cosine_score"].values.reshape(-1, 1)
        y_train = train_df["is_duplicate"]

        # Prepare test data
        X_test = test_df["cosine_score"].values.reshape(-1, 1)
        y_test = test_df["is_duplicate"]

        # Initialize the model
        model = LogisticRegression()

        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        model_eval = {
            metric: function(y_test, predictions)
            for metric, function in METRICS.items()
        }

        results[model_name] = model_eval

    return results


@stub.local_entrypoint()
def main():
    import json
    from tabulate import tabulate

    results = train_logistic_regression.remote()

    with open("output.json", "w") as f:
        json.dump(results, f, indent=2)
    values = []
    eval_metrics = [eval for eval in METRICS]
    eval_metrics.remove("AUC")
    eval_metrics.insert(0, "AUC")
    for model in results:
        model_evals = [model] + [results[model][eval] for eval in eval_metrics]
        values.append(model_evals)
    values.sort(key=lambda x: x[1], reverse=True)
    print(tabulate(values, ["Model Name", *eval_metrics]))
