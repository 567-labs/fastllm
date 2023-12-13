import optuna
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from dataset import PairsDataset, load_and_split_data, load_df_sentence_compression
from model import SimilarityModel
import pytorch_lightning as pl
import pathlib

model_id = "BAAI/bge-small-en-v1.5"


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
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        from sklearn.model_selection import StratifiedShuffleSplit
        import numpy as np

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


def objective(trial, checkpoint_dirpath):
    # Hyperparameters to be optimized
    embedding_size = 384
    dropout_fraction = 0.5
    n_dims = trial.suggest_int(
        "n_dims", low=embedding_size // 2, high=embedding_size * 3, log=True
    )
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_loguniform("lr", low=1e-5, high=1e-3)
    name = f"similarity-model-{n_dims}-{batch_size}"

    print(
        {
            "n_dims": n_dims,
            "batch_size": batch_size,
            "lr": lr,
        }
    )

    # Load and split data
    (
        train_df1,
        val_df1,
        test_df1,
        train_df2,
        val_df2,
        test_df2,
    ) = load_and_split_data(*load_df_sentence_compression())

    # Create Datasets and DataLoaders for training, validation, and test
    train_dataset = PairsDataset(train_df1, train_df2)
    val_dataset = PairsDataset(val_df1, val_df2)
    test_dataset = PairsDataset(test_df1, test_df2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # sampler=StratifiedSampler(
        #     class_vector=torch.tensor(train_target), batch_size=batch_size
        # ),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model
    model = SimilarityModel(
        embedding_size=embedding_size,
        n_dims=n_dims,
        dropout_fraction=dropout_fraction,
        lr=lr,
        use_relu=True,
    )

    # Define a checkpoint callback
    f1_check = ModelCheckpoint(
        monitor="val_f1",
        # TODO: rename checkpoint
        dirpath=checkpoint_dirpath,
        filename=f"checkpoint-{trial.number}",
        # filename=name + "-{epoch:02d}-{val_loss:.2f}-{val_recall:.2f}-{val_f1:.2f}",
        save_top_k=1,
        mode="max",
    )

    # Define TensorBoard logger
    logger = TensorBoardLogger("tb_stratified", name=name)

    # Log hyperparameters
    logger.log_hyperparams(
        {
            "embedding_size": embedding_size,
            "dropout_fraction": dropout_fraction,
            "batch_size": batch_size,
            "lr": lr,
            "n_dims": n_dims,
        }
    )

    # Initialize trainer with TensorBoard logger
    trainer = pl.Trainer(
        max_epochs=3, #TODO: up this for actual training
        logger=logger,
        callbacks=[f1_check],
    )

    # TODO: find somewhere to make it a modal stub around here to do parallel optuna stuff

    # Train the model and log validation evaluations at every epoch
    trainer.fit(model, train_loader, val_loader)

    # Evaluate on test set and return the loss
    test_results = trainer.test(model, test_loader)
    print(test_results[0].keys())
    test_loss = test_results[0]["test_f1"]

    return test_loss


def run_optuna(checkpoint_dirpath):
    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, checkpoint_dirpath), n_trials=10)

    # Print the result
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")

    return trial._trial_id

if __name__ == "__main__":
    checkpoints_dirpath = pathlib.Path("./checkpoints_stratified")
    best_trial_id = run_optuna(checkpoints_dirpath)
    print("--------------------------")
    print("best trial id:", best_trial_id)
    
