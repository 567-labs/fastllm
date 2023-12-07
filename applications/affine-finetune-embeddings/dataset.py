import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel


def load_df():
    # csv dataset
    # df = pd.read_csv("text_dataset.csv")

    # test dataset
    data = {
        "query": ["hello world", "hello world", "hello world"],
        "fact": ["hi world", "hi world", "hi world"]
    }
    df = pd.DataFrame(data)

    print(f"Loading {len(df)} rows of data")

    df1 = df["query"].tolist()
    df2 = df["fact"].tolist()

    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    return df1, df2


def load_and_split_data():
    df1, df2 = load_df()

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


class EmbeddingDataset(Dataset):
    def __init__(self, df1, df2, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        encoded_input_1 = self.tokenizer(
            df1[0].values.tolist(), padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input_2 = self.tokenizer(
            df2[0].values.tolist(), padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            model_output_1 = self.model(**encoded_input_1)
            model_output_2 = self.model(**encoded_input_2)
            # Perform pooling. In this case, cls pooling.
            self.embedding_1 = model_output_1[0][:, 0]
            self.embedding_2 = model_output_2[0][:, 0]

    def __len__(self):
        return len(self.embedding_1)

    def __getitem__(self, idx):
        return self.embedding_1[idx], self.embedding_2[idx]


if __name__ == "__main__":
    (
        train_df1,
        val_df1,
        test_df1,
        train_df2,
        val_df2,
        test_df2,
    ) = load_and_split_data()

    train_dataset = EmbeddingDataset(train_df1, train_df2, "BAAI/bge-small-en-v1.5")

    print(len(train_dataset[0][0]))
