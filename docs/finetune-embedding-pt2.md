# Fine-Tuning using Hyperparameter Optimization on Modal + Optuna

Fine-tuning using hyperparameter optimization (aka grid search) is a more advanced topic, however it can be very effective in tuning more performant models, including sentence transformer models. Modal is uniquely capable to do distributed hyperparameter optimizaton by using Modal's [NetworkFileSystem](https://modal.com/docs/guide/network-file-systems) feature with a hyperparameter framework, [Optuna](https://optuna.org/).

Basically, with Modal we can spin up dozens of serverless GPUs simulatneously to run dozens of hyperparameter tuning trials in parallel! Pretty cool, right?

## What is Optuna?

**TODO clarify how optuna works with trials**

Optuna is an open-source hyperparameter optimization framework designed for machine learning. It provides an efficient and easy-to-use interface for automatically searching for the best hyperparameters in your model training process. Specifically, how it works is it involves defining an objective function, in this case our loss, that is minimized for by testing a variety of hyperparameter combinations in various trials.

While this is a general technique used in machine learning, it is uniquely useful for fine-tuning sentence transformers as well. In addition to the standard hyperparameters we can optimize for the model, including learning rate, batch size, # of epochs, etc. it allows us to test some important hyperparameters specific to sentence transformers: the base embedding model and an optional linear layer.

Hyperparameters to test:

* Base Embedding Model: there are many different OSS embedding models and it's difficult to choose the right one. Testing multiple would help us find the most performant one for our specific task
* Optional Linear Layer Paramater Count: we can use a linear layer to help fine-tune our model and modify the output embedding dimension count using [sentence_transformers.models.Dense](https://www.sbert.net/docs/package_reference/models.html#sentence_transformers.models.Dense) This is useful for compressing our output embeddings to save space in a vector database
* Standard hyperparameters such as learning rate scheduler, batch size, # of epochs

## How do we use this with Modal?

**TODO link to repo**

Since the code for this is slightly more complicated than the simple fine-tuning we did previously, details can be found above.

The gist of it is that we are using Optuna with a [`JournalFileStorage`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalFileStorage.html#optuna.storages.JournalFileStorage) backend to store data from our multiple trials. `JournalFileStorage` allows us to use it with Network File System, or NFS.

Modal provides an [NFS resource](https://modal.com/docs/guide/network-file-systems) for us, which we instantiate with out stub. We then invoke multiple Modal Functions in parallel with Modal's `map()` ([details here](https://modal.com/docs/guide/scale)), provisioning GPU containers in parallel which act as workers for our distributed Optuna grid search. These workers test multiple trials in parallel
