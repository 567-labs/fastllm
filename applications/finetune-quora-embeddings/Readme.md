# Finetuning Quora Embeddings

This example shows how we can perform a hyper-parameter search using modal's GPUs. We've published a detailed writeup which walks you through the process [here](https://modal.com/blog/fine-tuning-embeddings)

## Description

To quickly get started, we've provided a few files that might be of interest

1. `main.py` : This showcases how you can use Modal to finetune a model given a set of hyper-parameters
2. `optimize_plain.py` : This showcases how we can perform a quick hyper-parameter search using Modal's containers.
3. `optimize.py` : This showcases how we can use `Optuna` to perform hyper-parameter search using Modal. This takes significantly longer since Optuna has a variety of different heuristics to optimize the pruning of hyper-parameters to find the optimal hyper-parameters.

## Getting Started

You'll need a few packages to get started - we recommend using a virtual environment to install all of the dependencies listed in the `requirements.txt`

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements
```

Once you've done so, you'll need to authenticate with Modal. To do so, run the command `modal token new`. Once you've done so, you can run any of the commands using `modal run <script name>.py`
