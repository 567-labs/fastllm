# Finetuning Embeddings on Wikipedia

This provides a simple example of how to download a dataset from Hugging Face and run a massively parallelizable embedding job on Modal.

## Volume

We need to first download our Wikipedia dataset. This will take approximately `450`s if we use the native `load_dataset` functionality from hugging Face. 

We can speed it up using the `num_proc` keyword

- `num_proc=1` : 450
- `num_proc=4`: 301
- `num_proc=10` : 333

So setting a number between 4-10 should help speed it up significantly.

## Embedding Script

For our next step, we'll


```bash
modal serve main.py
```