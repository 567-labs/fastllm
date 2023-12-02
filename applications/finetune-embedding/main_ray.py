import ray
import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ray import tune
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset, load_and_split_data
from model import SimilarityModel
from sklearn.model_selection import StratifiedShuffleSplit


class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    def __init__(self, class_vector, batch_size):
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def train_model(config, checkpoint_dir=None):
    embedding_size = 1536
    dropout_fraction = 0.5
    n_dims = int(config["n_dims"])
    batch_size = int(config["batch_size"])
    lr = config["lr"]
    use_relu = config["use_relu"]

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
    val_dataset = EmbeddingDataset(val_df1, val_df2, val_target)
    test_dataset = EmbeddingDataset(test_df1, test_df2, test_target)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=StratifiedSampler(
            class_vector=torch.tensor(train_target), batch_size=batch_size
        ),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SimilarityModel(
        embedding_size=embedding_size,
        n_dims=n_dims,
        dropout_fraction=dropout_fraction,
        lr=lr,
        use_relu=use_relu,
    )

    early_stop = EarlyStopping(monitor="val_auc", patience=50, mode="max", verbose=True)

    name = f"sim_d{n_dims}_b{batch_size}_r{use_relu}"

    auc = ModelCheckpoint(
        monitor="val_auc",
        dirpath=".",
        filename=name + "-{val_auc:.2f}-{epoch:02d}",
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(
        name=name,
        project="finetune-embedding-v5",
        config={
            "embedding_size": embedding_size,
            "dropout_fraction": dropout_fraction,
            "batch_size": batch_size,
            "lr": lr,
            "n_dims": n_dims,
            "use_relu": use_relu,
        },
    )

    trainer = pl.Trainer(
        max_epochs=400,
        logger=[wandb_logger],
        callbacks=[early_stop, auc],
    )

    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model, test_loader)

    tune.report(
        auc=test_results[0]["test_auc"],
        recall=test_results[0]["test_recall"],
        precision=test_results[0]["test_precision"],
        f1=test_results[0]["test_f1"],
        loss=test_results[0]["test_loss"],
    )


embedding_size = 1536

config = {
    "n_dims": tune.uniform(embedding_size // 2, embedding_size * 2),
    "batch_size": tune.choice([64, 128]),
    "lr": tune.loguniform(1e-5, 1e-3),
    "use_relu": tune.choice([True, False]),
}

analysis = ray.tune.run(
    train_model,
    config=config,
    metric="auc",
    mode="max",
    num_samples=40,
    local_dir="./ray_results",
)

print(f"Best trial config: {analysis.best_config}")
print(f"Best trial final metric score: {analysis.best_result['auc']}")
