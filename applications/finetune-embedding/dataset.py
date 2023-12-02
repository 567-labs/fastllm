import torch
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split
import pandas as pd
import torch


def load_df():
    df = pd.read_csv("~/Downloads/embedding_dataset.csv")
    df["similarity"] = (df["relevancy_tag"] == "RELEVANT").astype(int)

    print(f"Loading {len(df)} rows of data")

    def safe_literal_eval(x):
        try:
            return eval(x)
        except Exception:
            return None

    df["query_embedding"] = df["query_embedding"].apply(safe_literal_eval)
    df["fact_embedding"] = df["fact_embedding"].apply(safe_literal_eval)

    # Drop rows with None (malformed data) if needed
    df.dropna(subset=["query_embedding", "fact_embedding"], inplace=True)

    df1 = df["query_embedding"].tolist()
    df2 = df["fact_embedding"].tolist()

    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    target_similarity = df["similarity"].tolist()
    return df1, df2, target_similarity


def load_and_split_data():
    df1, df2, target_similarity = load_df()

    # Split data into training and temporary sets (80% training, 20% temporary)
    (
        train_df1,
        temp_df1,
        train_df2,
        temp_df2,
        train_target,
        temp_target,
    ) = train_test_split(
        df1,
        df2,
        target_similarity,
        stratify=target_similarity,
        test_size=0.4,
        random_state=42,
    )

    # Split temporary set into validation and test sets (50% validation, 50% test)
    val_df1, test_df1, val_df2, test_df2, val_target, test_target = train_test_split(
        temp_df1, temp_df2, temp_target, test_size=0.5, random_state=42
    )

    return (
        train_df1,
        val_df1,
        test_df1,
        train_df2,
        val_df2,
        test_df2,
        train_target,
        val_target,
        test_target,
    )


class EmbeddingDataset(Dataset):
    def __init__(self, df1, df2, target_similarity):
        self.embedding_1 = torch.tensor(df1.values, dtype=torch.float32)
        self.embedding_2 = torch.tensor(df2.values, dtype=torch.float32)
        self.target_similarity = torch.tensor(target_similarity)

    def __len__(self):
        return len(self.embedding_1)

    def __getitem__(self, idx):
        return self.embedding_1[idx], self.embedding_2[idx], self.target_similarity[idx]


if __name__ == "__main__":
    (
        train_df1,
        val_df1,
        test_df1,
        train_df2,
        val_df2,
        test_df2,
        train_target,
        val_target,
        test_target,
    ) = load_and_split_data()

    train_dataset = EmbeddingDataset(train_df1, train_df2, train_target)

    print(train_dataset[0])
