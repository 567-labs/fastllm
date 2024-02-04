import itertools
from typing import List
from modal import Image, Stub, Volume, Secret, gpu
import os

GPU_CONFIG = gpu.A100()

# Dataset Configuration
DATASET_VOLUME = Volume.persisted("datasets")
DATASET_DIR = "/data"
COSINE_SIMILARITY_DIR = f"{DATASET_DIR}/cosine-similarity"


stub = Stub("process-embeddings")


image = Image.debian_slim().pip_install(
    "scikit-learn",
    "sentence_transformers",
    "pyarrow",
    "numpy",
    "torch",
    "pandas",
    "matplotlib",
    "tqdm",
)


@stub.function(image=image, volumes={DATASET_DIR: DATASET_VOLUME}, timeout=86400)
def generate_cosine_similarity_scores():
    import glob
    import numpy as np
    import pyarrow as pa
    from sentence_transformers import util
    import torch
    from pandas import DataFrame

    # Get all .arrow files in the /cached-embeddings directory
    arrow_files = glob.glob(f"{DATASET_DIR}/cached-embeddings/*.arrow")
    # arrow_files = ["/data/cached-embeddings/text-embedding-3-small-test.arrow"]

    # Create the /cosine-similarity directory if it doesn't exist
    if not os.path.exists(COSINE_SIMILARITY_DIR):
        os.makedirs(COSINE_SIMILARITY_DIR)
    print(arrow_files)
    # Copy each .arrow file to the /cosine-similarity directory
    for file in arrow_files:
        new_file_name = os.path.join(
            COSINE_SIMILARITY_DIR,
            os.path.basename(file).replace(".arrow", "-cossim.arrow"),
        )
        if os.path.exists(new_file_name):
            print(f"Skipping {file} since it has already been processed")
            continue
        # Load the original arrow file
        table = pa.ipc.open_file(file).read_all()
        df: DataFrame = table.to_pandas()

        print(f"Number of rows in the dataset: {len(df)}")

        # Extract embeddings and labels directly using pandas DataFrame methods
        embeddings_sentence_1 = np.array(
            df["questions"].apply(lambda x: x["embeddings"][0]).tolist()
        )
        embeddings_sentence_2 = np.array(
            df["questions"].apply(lambda x: x["embeddings"][1]).tolist()
        )
        dataset_labels = df["is_duplicate"].tolist()

        from tqdm import tqdm

        batch_size = 2000
        num_batches = int(np.ceil(len(embeddings_sentence_1) / batch_size))
        predictions = []

        for i in tqdm(range(num_batches), desc="Processing cosine scores"):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(embeddings_sentence_1))

            batch_embeddings_sentence_1 = embeddings_sentence_1[start_index:end_index]
            batch_embeddings_sentence_2 = embeddings_sentence_2[start_index:end_index]

            cosine_scores = util.cos_sim(
                torch.Tensor(batch_embeddings_sentence_1),
                torch.Tensor(batch_embeddings_sentence_2),
            )
            batch_predictions = np.diag(cosine_scores.cpu()).tolist()
            predictions.extend(batch_predictions)

        # Create a new table with cosine_similarity and is_duplicate columns
        new_table = pa.Table.from_arrays(
            [
                pa.array(predictions),
                pa.array(dataset_labels),
            ],
            names=[
                "cosine_score",
                "is_duplicate",
            ],
        )

        # Save the new table as an arrow file in the /cosine-similarity directory

        with pa.OSFile(new_file_name, "wb") as sink:
            writer = pa.RecordBatchFileWriter(sink, new_table.schema)
            writer.write_table(new_table)
            writer.close()

        print(f"Generated new file of {new_file_name}")
        DATASET_VOLUME.commit()


@stub.function(
    image=image, volumes={DATASET_DIR: DATASET_VOLUME}, gpu=GPU_CONFIG, timeout=86400
)
def generate_visualisation():
    import glob
    import io
    import pyarrow as pa
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    import textwrap

    # Get all .arrow files in the /cached-embeddings directory
    arrow_files = glob.glob(f"{DATASET_DIR}/cosine-similarity/*.arrow")

    res = []
    # Copy each .arrow file to the /cosine-similarity directory
    for file in arrow_files:
        table = pa.ipc.open_file(file).read_all()
        df: DataFrame = table.to_pandas()

        df_label_1 = df[df["is_duplicate"] == 1]["cosine_score"]
        df_label_0 = df[df["is_duplicate"] == 0]["cosine_score"]

        model_name = os.path.splitext(os.path.basename(file))[0].replace("-cossim", "")
        # Create the plot
        plt.figure()
        plt.hist(df_label_1, alpha=0.5, label="Similar", bins=50)
        plt.hist(df_label_0, alpha=0.5, label="Dissimilar", bins=50)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.1, left=0.15, right=0.75)
        title = f"Distribution of Cosine Similarity Scores for {model_name}"
        plt.title("\n".join(textwrap.wrap(title, 40)), loc="left", pad=10)
        plt.xlabel("Cosine Similarity Score")
        plt.ylabel("Frequency")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)  # Rewind the buffer
        image_bytes = buf.getvalue()
        buf.close()

        res.append([model_name, image_bytes])

    return res


@stub.local_entrypoint()
def main():
    import os

    # generate_cosine_similarity_scores.remote()

    img_dir = "./img"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for model_name, image_bytes in generate_visualisation.remote():
        with open(f"./img/{model_name}-distribution.png", "wb") as f:
            f.write(image_bytes)
