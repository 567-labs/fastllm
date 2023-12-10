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
    for item in dataset[:5000]["set"]:
    # for item in dataset[:]["set"]:
        l1.append(item[0])
        l2.append(item[1])
    df1 = pd.DataFrame(l1)
    df2 = pd.DataFrame(l2)
    return df1, df2

def load_df_sample():
    # csv dataset
    df = pd.read_csv("text_dataset.csv")

    # test dataset
    # data = {
    #     "query": ["hello world", "hello world", "hello world"],
    #     "fact": ["hi world", "hi world", "hi world"]
    # }
    # df = pd.DataFrame(data)

    print(f"Loading {len(df)} rows of data")

    df1 = df["query"].tolist()
    df2 = df["fact"].tolist()

    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    print(df1)
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


class EmbeddingDataset(Dataset):
    def __init__(self, df1, df2, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.df1 = df1
        self.df2 = df2
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)

        
    def __len__(self):
        return len(self.df1)

    def __getitem__(self, idx):
        # encoded_input_1 = self.tokenizer(
        #     self.df1[0].values.tolist(), padding=True, truncation=True, return_tensors="pt"
        # ).to(self.device)
        # encoded_input_2 = self.tokenizer(
        #     self.df2[0].values.tolist(), padding=True, truncation=True, return_tensors="pt"
        # ).to(self.device)

        # Tokenize only the sentence at the given index
        encoded_input_1 = self.tokenizer(
            self.df1.iat[idx,0], padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input_2 = self.tokenizer(
            self.df2.iat[idx,0], padding=True, truncation=True, return_tensors="pt"
        )

        # Move the encoded inputs to the GPU
        encoded_input_1 = {k: v.to(self.device) for k, v in encoded_input_1.items()}
        encoded_input_2 = {k: v.to(self.device) for k, v in encoded_input_2.items()}

        with torch.no_grad():
            model_output_1 = self.model(**encoded_input_1)
            model_output_2 = self.model(**encoded_input_2)
            # Perform pooling. In this case, cls pooling.
            self.embedding_1 = model_output_1[0][:, 0]
            self.embedding_2 = model_output_2[0][:, 0]

            self.embedding_1 = torch.nn.functional.normalize(self.embedding_1, p=2, dim=1)
            self.embedding_2 = torch.nn.functional.normalize(self.embedding_2, p=2, dim=1)

        return self.embedding_1[0], self.embedding_2[0]


if __name__ == "__main__":
    (
        train_df1,
        val_df1,
        test_df1,
        train_df2,
        val_df2,
        test_df2,
    ) = load_and_split_data(*load_df_sentence_compression())

    train_dataset = EmbeddingDataset(train_df1, train_df2, "BAAI/bge-small-en-v1.5")

    train_dataset[0]
    # print("first row", train_dataset[0])
    # print("embedding size", len(train_dataset[0][0]))
