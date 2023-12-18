# Sentence Transformers Finetuning

Fine tune a sentence transformer model on a dataset with modal.

## dev stuff

Stuff I plan to do:

* I will do a more complete training run using the entire dataset tomorrow, and will report the accuracy changes then.
* grid search with optuna + modal
* improve the evaluation step from just printing out the metrics. Possibly save it somewhere and make a basic matplotlib graph of it. also research which specific metrics it outputs are relevant to us.

## Overview

Fine tune `BAAI/bge-small-en-v1.5` on the [Quora Pairs Dataset](https://huggingface.co/datasets/quora) using [Sentence Transformers](https://www.sbert.net/).

Features:

* based on this article: <https://www.sbert.net/examples/training/quora_duplicate_questions/README.html>
* uses Modal for training
  * stores checkpoints and output model into a persistent modal volume (WARNING!! delete the persistent volume after you're done using it since it costs MONEY $$$ to store)

## Instructions

Run locally:

```bash
python main.py
```

Run on Modal:

```bash
modal run main.py
```

## Sample Output

Currently, this code evaluates the model beforehand, trains the model, and then evaluates it afterwards.

The following is an example output of the model after running on Modal. (note that I set extremely low hyperparemters for the sake of development i.e. using 2% of the dataset, low epochs, low batch size. I will use reasonable hyperparameters when training later)

```bash
pre train eval {'cossim': {'accuracy': 0.7778327560613557, 'accuracy_threshold': 0.8726286888122559, 'f1': 0.7237057220708447, 'f1_threshold': 0.8222078680992126, 'precision': 0.6097337006427915, 'recall': 0.8900804289544236, 'ap': 0.7658146652992561}, 'manhattan': {'accuracy': 0.7758535378525483, 'accuracy_threshold': 7.753220558166504, 'f1': 0.7228915662650602, 'f1_threshold': 9.253597259521484, 'precision': 0.6111111111111112, 'recall': 0.8847184986595175, 'ap': 0.7655460401099523}, 'euclidean': {'accuracy': 0.7778327560613557, 'accuracy_threshold': 0.5047200918197632, 'f1': 0.7237057220708447, 'f1_threshold': 0.596308708190918, 'precision': 0.6097337006427915, 'recall': 0.8900804289544236, 'ap': 0.7658146652992562}, 'dot': {'accuracy': 0.7778327560613557, 'accuracy_threshold': 0.8726286888122559, 'f1': 0.7237057220708447, 'f1_threshold': 0.8222079277038574, 'precision': 0.6097337006427915, 'recall': 0.8900804289544236, 'ap': 0.7658152473425966}}
```

```bash
...truncated...
Iteration:  89%|████████▉ | 374/420 [00:20<00:02, 18.33it/s]
Iteration:  90%|████████▉ | 376/420 [00:20<00:02, 18.36it/s]
Iteration:  90%|█████████ | 378/420 [00:20<00:02, 18.35it/s]
Iteration:  90%|█████████ | 380/420 [00:21<00:02, 18.37it/s]
Iteration:  91%|█████████▏| 384/420 [00:21<00:01, 18.40it/s]
Iteration:  92%|█████████▏| 386/420 [00:21<00:01, 18.36it/s]
Iteration:  93%|█████████▎| 390/420 [00:21<00:01, 18.29it/s]
Iteration:  93%|█████████▎| 392/420 [00:21<00:01, 18.22it/s]
Iteration:  94%|█████████▍| 394/420 [00:21<00:01, 18.25it/s]
Iteration:  94%|█████████▍| 396/420 [00:21<00:01, 18.23it/s]
Iteration:  95%|█████████▍| 398/420 [00:22<00:01, 18.17it/s]
Iteration:  95%|█████████▌| 400/420 [00:22<00:01, 18.22it/s]
Iteration:  96%|█████████▌| 402/420 [00:22<00:00, 18.24it/s]
Iteration:  96%|█████████▌| 404/420 [00:22<00:00, 18.16it/s]
Iteration:  97%|█████████▋| 407/420 [00:22<00:00, 18.28it/s]
Iteration:  97%|█████████▋| 409/420 [00:22<00:00, 18.25it/s]
Iteration:  98%|█████████▊| 411/420 [00:22<00:00, 18.19it/s]
Iteration: 100%|██████████| 420/420 [00:23<00:00, 18.06it/s]
[A
Epoch: 100%|██████████| 10/10 [04:01<00:00, 24.19s/it]
...truncated...
```

```bash
post train eval {'cossim': {'accuracy': 0.7847600197921821, 'accuracy_threshold': 0.8737390637397766, 'f1': 0.7327188940092166, 'f1_threshold': 0.7969428896903992, 'precision': 0.6424242424242425, 'recall': 0.8525469168900804, 'ap': 0.7675069279956492}, 'manhattan': {'accuracy': 0.7832756061355765, 'accuracy_threshold': 7.995540618896484, 'f1': 0.7316810344827586, 'f1_threshold': 10.891837120056152, 'precision': 0.6117117117117117, 'recall': 0.9101876675603218, 'ap': 0.7675313603430923}, 'euclidean': {'accuracy': 0.7847600197921821, 'accuracy_threshold': 0.5025155544281006, 'f1': 0.7327188940092166, 'f1_threshold': 0.6372709274291992, 'precision': 0.6424242424242425, 'recall': 0.8525469168900804, 'ap': 0.7675062483471642}, 'dot': {'accuracy': 0.7847600197921821, 'accuracy_threshold': 0.8737390041351318, 'f1': 0.7327188940092166, 'f1_threshold': 0.796942949295044, 'precision': 0.6424242424242425, 'recall': 0.8525469168900804, 'ap': 0.7675062483471642}}
```
