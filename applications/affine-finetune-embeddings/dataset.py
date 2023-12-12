import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

from datasets import load_dataset


def load_df_sentence_compression() -> (pd.DataFrame, pd.DataFrame):
    # embedding-data/sentence-compression only has a "train" split
    dataset = load_dataset("embedding-data/sentence-compression", split="train")

    # probably a faster way to do this with huggingface dataset library
    l1 = []
    l2 = []
    for item in dataset[:1500]["set"]:
        # for item in dataset[:]["set"]:
        l1.append(item[0])
        l2.append(item[1])
    df1 = pd.DataFrame(l1)
    df2 = pd.DataFrame(l2)
    return df1, df2


def load_and_split_data(df1, df2):
    # df1: query
    # df2: fact
    # Split data into training and temporary sets (80% training, 20% temporary)
    (
        train_df1,
        temp_df1,
        train_df2,
        temp_df2,
    ) = train_test_split(
        df1,
        df2,
        test_size=0.4,
        random_state=42,
    )

    # Split temporary set into validation and test sets (50% validation, 50% test)
    (
        val_df1,
        test_df1,
        val_df2,
        test_df2,
    ) = train_test_split(temp_df1, temp_df2, test_size=0.5, random_state=42)

    return (
        train_df1,
        val_df1,
        test_df1,
        train_df2,
        val_df2,
        test_df2,
    )


class PairsDataset(Dataset):
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

    def __len__(self):
        return len(self.df1)

    def __getitem__(self, idx):
        return self.df1.iat[idx, 0], self.df2.iat[idx, 0]


if __name__ == "__main__":
    (
        train_df1,
        val_df1,
        test_df1,
        train_df2,
        val_df2,
        test_df2,
    ) = load_and_split_data(*load_df_sentence_compression())

    train_dataset = PairsDataset(train_df1, train_df2)

    print(train_dataset[0])
    # print("first row", train_dataset[0])
    # print("embedding size", len(train_dataset[0][0]))
