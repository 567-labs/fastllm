# Finetuning Embeddings on Wikipedia

This provides a simple example of how to download a dataset from Hugging Face and run a massively parallelizable embedding job on Modal.

## Volume

We need to first download our Wikipedia dataset. This will take approximately `450`s if we use the native `load_dataset` functionality from hugging Face.


```bash
modal serve main.py
```