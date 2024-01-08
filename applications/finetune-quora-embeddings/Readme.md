# Finetuning Quora Embeddings

This example shows how we can finetune an open source model from hugging face on our own dataset to improve performance drastically. We've published a detailed writeup which walks you through the implementation here.

## Description

There is currently 1 file in this repository

1. `main.py`: This showcases how to use Modal to download a hugging face dataset and then subsequently use it to finetune a model of your choice.

## Getting Started

You'll need a few packages to get started - we recommend using a virtual environment to install all of the dependencies listed in the `requirements.txt`

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements
```

Once you've done so, you'll need to authenticate with Modal. To do so, run the command `modal token new`. 

This will open up a new tab in your default browser and allow you to run, deploy and configure all of your Modal applications from your terminal.

## Finetuning our model

> We've configured wandb support for this specific script - so make sure to have added wandb in your `Modal` secrets

Now that we've downloaded our wikipedia dataset, we can now embed the entire dataset using our `main.py` script. We can run it using the command 

```
modal run main.py
```

This will kick start a fine tuning job and load the `Quora` dataset into a volume and save it. Each epoch run will take ~6-15 minutes. 