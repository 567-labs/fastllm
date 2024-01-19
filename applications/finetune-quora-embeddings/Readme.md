# Finetuning Quora Embeddings

This example shows how we can finetune an open source model from hugging face on our own dataset to improve performance drastically. We've published a detailed writeup which walks you through the implementation here.

## Description

There are currently two files in this repository

1. `finetune.py`: This showcases how to use Modal to download a hugging face dataset and then subsequently use it to finetune a model of your choice.
2. `benchmark.py`: This showcases how to benchmark a model against a dataset - in our case we have chosen a cleaned dataset with pre-defined splits that minimise test-time leakage

## Getting Started

You'll need a few packages to get started - we recommend using a virtual environment to install all of the dependencies listed in the `requirements.txt`

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements
```

Once you've done so, you'll need to authenticate with Modal. To do so, run the command `modal token new`. 

This will open up a new tab in your default browser and allow you to run, deploy and configure all of your Modal applications from your terminal.

## Benchmarking the Model

> We've configured the OpenAI and Cohere client sdks in our benchmarking script. To make sure it works, you'll need to have added OpenAI and Cohere in our `Modal` secrets. 
> 
> You'll need the following enviroment variables 
> - COHERE_API_KEY
> - OPENAI_API_KEY
> ________

You can configure the OSS models tested by modifying the `models` variable in the `local_entrypoint` in the stub ( See below for an example )

```py
@stub.local_entrypoint()
def main():
    from tabulate import tabulate

    generate_dataset_split.remote()

    res = {}
    res["text-embeddings-ada-v2"] = benchmark_openai.remote()
    res["embed-multilingual-v3.0"] = benchmark_cohere_roc.remote()
    models = [
        "llmrails/ember-v1",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-large",
        "infgrad/stella-base-en-v2",
        "sentence-transformers/gtr-t5-large",
```

Once you've configured the necessary secrets and models to benchmark against, you can run the script by running the command 

```
modal run benchmark.py
```

In our case, we obtained the following results when we ran the script

| Model Name                         |      AUC    |
|------------------------------------|-------------|
| sentence-transformers/gtr-t5-large | 0.93892     |
| embed-multilingual-v3.0            | 0.938904    |
| llmrails/ember-v1                  | 0.937499    |
| infgrad/stella-base-en-v2          | 0.934832    |
| BAAI/bge-base-en-v1.5              | 0.931893    |
| thenlper/gte-large                 | 0.93085     |
| text-embeddings-ada-v2             | 0.928656    |

Note that your results might vary slightly.

## Finetuning our model

> We've configured wandb support for this specific script - so make sure to have added wandb in your `Modal` secrets

Now that we've downloaded our wikipedia dataset, we can now embed the entire dataset using our `main.py` script. We can run it using the command 

```
modal run main.py
```

This will kick start a fine tuning job and load the `Quora` dataset into a volume and save it. Each epoch run will take ~6-15 minutes. 