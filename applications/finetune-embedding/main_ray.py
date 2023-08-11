from modal import Image, Stub, Mount
from dataset import EmbeddingDataset, load_and_split_data
from model import SimilarityModel
from sampler import StratifiedSampler

image = Image.from_dockerhub("nvcr.io/nvidia/pytorch:22.12-py3").pip_install(
    "wandb",
    "ray",
    "pytorch-lightning",
    "torchmetrics",
    "scikit-learn",
    "pandas",
    "torch",
    "torchvision",
)

stub = Stub("embedding-finetune", image=image)


def train_model(
    n_dims: int, batch_size: int, lr: float, drop_bad_routers: bool = False
):
    import torch

    import pytorch_lightning as pl
    from pytorch_lightning.loggers.wandb import WandbLogger
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        TQDMProgressBar,
    )

    from torch.utils.data import DataLoader
    import wandb

    wandb.login(key="0948242e8fc4b9886a127af664271c117743439a")

    embedding_size = 1536
    dropout_fraction = 0.5

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
    ) = load_and_split_data(drop_bad_routers=drop_bad_routers)

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
        use_relu=False,
    )

    early_stop = EarlyStopping(monitor="val_auc", patience=50, mode="max", verbose=True)

    name = f"sim_d:{n_dims}_b:{batch_size}_filter:{drop_bad_routers}"

    auc = ModelCheckpoint(
        monitor="val_auc",
        filename=name,
        save_top_k=1,
        mode="max",
        save_weights_only=True,
        every_n_epochs=5,
    )

    wandb_logger = WandbLogger(
        name=name,
        project="relevance-embedding",
        log_model="all",
        config={
            "embedding_size": embedding_size,
            "batch_size": batch_size,
            "lr": lr,
            "n_dims": n_dims,
            "drop_bad_routers": drop_bad_routers,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
        },
    )

    loading_bar = TQDMProgressBar(refresh_rate=0)

    trainer = pl.Trainer(
        max_epochs=400,
        logger=[wandb_logger],
        callbacks=[early_stop, auc, loading_bar],
    )

    trainer.fit(model, train_loader, val_loader)
    resp = trainer.test(model, test_loader)
    return {
        "auc": resp[0]["test_auc"],
        "recall": resp[0]["test_recall"],
        "precision": resp[0]["test_precision"],
        "f1": resp[0]["test_f1"],
    }


@stub.function(
    gpu="A100",
    timeout=86400,
    mounts=[
        Mount.from_local_dir(
            "data",
            remote_path="/root/data",
        )
    ],
)
def tune(n_samples: int):
    import ray
    from ray import tune

    ray.init(num_gpus=1)

    def train_from_params(config):
        n_dims = int(config["n_dims"])
        batch_size = int(config["batch_size"])
        lr = config["lr"]
        drop_bad_routers = config["drop_bad_routers"]
        test_results = train_model(n_dims, batch_size, lr, drop_bad_routers)
        tune.report(
            **test_results,
        )

    config = {
        "n_dims": tune.uniform(500, 3000),
        "batch_size": tune.choice([32, 64, 96, 128]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "drop_bad_routers": tune.choice([True, False]),
    }

    analysis = ray.tune.run(
        train_from_params,
        config=config,
        metric="auc",
        mode="max",
        num_samples=n_samples,
    )
    return analysis.best_config, analysis.best_result["auc"]


@stub.local_entrypoint()
def finetune(n_samples: int):
    best_config, best_auc = tune.call(n_samples=n_samples)
    print("Best config is:", best_config)
    print("Best AUC is:", best_auc)
