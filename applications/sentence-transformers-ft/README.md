# Sentence Transformers Finetuning

Fine tune a sentence transformer model on a dataset with modal.

## Overview

We fine tune `BAAI/bge-small-en-v1.5` on the [Quora Pairs Dataset](https://huggingface.co/datasets/quora) using [Sentence Transformers](https://www.sbert.net/). Explanations for the loss function, evaluation, and dataset chosen can be found in more detail here: <https://www.sbert.net/examples/training/quora_duplicate_questions/README.html>.

In summary, the Quora Pairs Dataset contains pairs of texts that are marked as either duplicates or not duplicates.

```json
{
    "is_duplicate": true,
    "questions": {
        "id": [1, 2],
        "text": ["Is this a sample question?", "Is this an example question?"]
    }
}
```

Finetuning embedding models can have different goals, ranging from improved performance in information retrieval to better accuracy when identifying semantically duplicate texts. In this example, we fine tune our sentence transformer to better identify thes duplicate texts, i.e. the fine-tuned embeddings that is produces are closer to each other if they are semantically duplicates. This has applications in not only classifying if a pair of texts are semantically similar, but also mining duplicate questions in large corpuses of data. Other ways of methods of fine-tuning can optimize for different goals, such as information retrieval. (possibly add more information about the dataset format, loss, and evaluation used for other fine-tuning goals).

We use Modal for training our fine tuned sentence model. The checkpoints, output model, and evaluation metrics are then stored in a Modal volume for easy access by the user. The stored model is Huggingface compatible, and also we can use our huggingface.
NOTE: stores checkpoints and output model into a persistent modal volume (WARNING!! delete the persistent volume after you're done using it since it costs MONEY $$$ to store)

## Results

I ran the script on the full quora pairs dataset using epoch=10, batch_size=32, loss=OnlineContrastiveLoss, and the evaluation as BinaryClassificationEvaluator using `BAAI/bge-small-en-v1.5` as the base embedding model.

The overall accuracy (measured by highest [AP score](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/BinaryClassificationEvaluator.py#L86-L87) in each section) increased from `0.7723` before training to `0.9156` after training. Detailed evals metrics are listed below. Also, note that this naively takes the post-training model and evaluates it, it doesn't take the best performing model and evaluate it yet, which can be changed. (can describe more about what the metrics mean)

Pre train:

```csv
epoch,steps,cossim_accuracy,cossim_accuracy_threshold,cossim_f1,cossim_precision,cossim_recall,cossim_f1_threshold,cossim_ap,manhattan_accuracy,manhattan_accuracy_threshold,manhattan_f1,manhattan_precision,manhattan_recall,manhattan_f1_threshold,manhattan_ap,euclidean_accuracy,euclidean_accuracy_threshold,euclidean_f1,euclidean_precision,euclidean_recall,euclidean_f1_threshold,euclidean_ap,dot_accuracy,dot_accuracy_threshold,dot_f1,dot_precision,dot_recall,dot_f1_threshold,dot_ap
-1,-1,0.7856736501026491,0.8703066110610962,0.7307950010196055,0.6417498081350729,0.8485319983764037,0.8361053466796875,0.7723541732429098,0.78485740433847,7.947874546051025,0.7303860890067786,0.6471171923960727,0.8382492220267893,8.820365905761719,0.7716763340290338,0.7856736501026491,0.509300172328949,0.7307950010196055,0.6417498081350729,0.8485319983764037,0.5725288391113281,0.7722899292609772,0.7856736501026491,0.8703066110610962,0.7307950010196055,0.6417498081350729,0.8485319983764037,0.8361053466796875,0.7723768728440874
```

Post train:

```csv
epoch,steps,cossim_accuracy,cossim_accuracy_threshold,cossim_f1,cossim_precision,cossim_recall,cossim_f1_threshold,cossim_ap,manhattan_accuracy,manhattan_accuracy_threshold,manhattan_f1,manhattan_precision,manhattan_recall,manhattan_f1_threshold,manhattan_ap,euclidean_accuracy,euclidean_accuracy_threshold,euclidean_f1,euclidean_precision,euclidean_recall,euclidean_f1_threshold,euclidean_ap,dot_accuracy,dot_accuracy_threshold,dot_f1,dot_precision,dot_recall,dot_f1_threshold,dot_ap
-1,-1,0.9041282247891365,0.8691110610961914,0.8722775812262652,0.8383977038747115,0.9090109592747937,0.8512338399887085,0.915680961336939,0.9042271636696431,8.107221603393555,0.872209640892915,0.8358134920634921,0.9119199025842241,8.558991432189941,0.9154704007309182,0.9041282247891365,0.511642336845398,0.8722775812262652,0.8383977038747115,0.9090109592747937,0.5454652309417725,0.9156165190275904,0.9041282247891365,0.8691110610961914,0.8722775812262652,0.8383977038747115,0.9090109592747937,0.8512338399887085,0.9156458045049147
```

## Instructions

Run locally:

```bash
python main.py
```

Run on Modal:

```bash
modal run main.py
```

Getting the training output volume:
A persistent volume will be created in Modal which can be found under <https://modal.com/YOUR_USERNAME/storage>. It will be stored in the format `sentence-transformers-ft-<UNIX_TIMESTAMP>`.
It can be retrieved to your local machine using the modal [volume CLI](https://modal.com/docs/reference/cli/volume).

View the contents of the volume

```bash
✗ modal volume ls sentence-transformers-ft-1702914571

Directory listing of '/' in 'sentence-transformers-ft-1702914571'
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ filename                                                ┃ type ┃ created/modified          ┃ size   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ BAAI--bge-small-en-v1.5-ft                              │ dir  │ 2023-12-18 11:04:40-05:00 │ 198 B  │
│ checkpoints                                             │ dir  │ 2023-12-18 12:59:51-05:00 │ 30 B   │
│ binary_classification_evaluation_post_train_results.csv │ file │ 2023-12-18 13:00:15-05:00 │ 1020 B │
│ binary_classification_evaluation_pre_train_results.csv  │ file │ 2023-12-18 10:51:45-05:00 │ 1020 B │
└─────────────────────────────────────────────────────────┴──────┴───────────────────────────┴────────┘
```

Create a directory and download the volume

```bash
✗ mkdir dump
✗ modal volume get sentence-transformers-ft-1702914571 "**" ./dump --force
Wrote 349 bytes to dump/checkpoints/113710/modules.json
Wrote 231508 bytes to dump/checkpoints/113710/vocab.txt
Wrote 695 bytes to dump/checkpoints/113710/special_tokens_map.json
Wrote 52 bytes to dump/checkpoints/113710/sentence_bert_config.json
Wrote 1020 bytes to dump/binary_classification_evaluation_post_train_results.csv
Wrote 124 bytes to dump/checkpoints/113710/config_sentence_transformers.json
Wrote 695 bytes to dump/BAAI--bge-small-en-v1.5-ft/special_tokens_map.json
Wrote 711649 bytes to dump/BAAI--bge-small-en-v1.5-ft/tokenizer.json
Wrote 231508 bytes to dump/BAAI--bge-small-en-v1.5-ft/vocab.txt
Wrote 711649 bytes to dump/checkpoints/113710/tokenizer.json
Wrote 748 bytes to dump/checkpoints/113710/config.json
Wrote 52 bytes to dump/BAAI--bge-small-en-v1.5-ft/sentence_bert_config.json
Wrote 349 bytes to dump/BAAI--bge-small-en-v1.5-ft/modules.json
Wrote 124 bytes to dump/BAAI--bge-small-en-v1.5-ft/config_sentence_transformers.json
Wrote 748 bytes to dump/BAAI--bge-small-en-v1.5-ft/config.json
Wrote 1242 bytes to dump/checkpoints/113710/tokenizer_config.json
Wrote 231508 bytes to dump/checkpoints/113500/vocab.txt
Wrote 1242 bytes to dump/BAAI--bge-small-en-v1.5-ft/tokenizer_config.json
Wrote 190 bytes to dump/BAAI--bge-small-en-v1.5-ft/1_Pooling/config.json
Wrote 711649 bytes to dump/checkpoints/113500/tokenizer.json
Wrote 695 bytes to dump/checkpoints/113500/special_tokens_map.json
Wrote 190 bytes to dump/checkpoints/113710/1_Pooling/config.json
Wrote 52 bytes to dump/checkpoints/113500/sentence_bert_config.json
Wrote 349 bytes to dump/checkpoints/113500/modules.json
Wrote 2345 bytes to dump/checkpoints/113710/README.md
Wrote 2345 bytes to dump/BAAI--bge-small-en-v1.5-ft/README.md
Wrote 124 bytes to dump/checkpoints/113500/config_sentence_transformers.json
Wrote 748 bytes to dump/checkpoints/113500/config.json
Wrote 190 bytes to dump/checkpoints/113500/1_Pooling/config.json
Wrote 231508 bytes to dump/checkpoints/113000/vocab.txt
Wrote 711649 bytes to dump/checkpoints/113000/tokenizer.json
Wrote 695 bytes to dump/checkpoints/113000/special_tokens_map.json
Wrote 52 bytes to dump/checkpoints/113000/sentence_bert_config.json
Wrote 349 bytes to dump/checkpoints/113000/modules.json
Wrote 1242 bytes to dump/checkpoints/113000/tokenizer_config.json
Wrote 1242 bytes to dump/checkpoints/113500/tokenizer_config.json
Wrote 5804 bytes to dump/BAAI--bge-small-en-v1.5-ft/eval/binary_classification_evaluation_results.csv
Wrote 190 bytes to dump/checkpoints/113000/1_Pooling/config.json
Wrote 748 bytes to dump/checkpoints/113000/config.json
Wrote 231508 bytes to dump/checkpoints/112500/vocab.txt
Wrote 1242 bytes to dump/checkpoints/112500/tokenizer_config.json
Wrote 711649 bytes to dump/checkpoints/112500/tokenizer.json
Wrote 695 bytes to dump/checkpoints/112500/special_tokens_map.json
Wrote 124 bytes to dump/checkpoints/113000/config_sentence_transformers.json
Wrote 349 bytes to dump/checkpoints/112500/modules.json
Wrote 124 bytes to dump/checkpoints/112500/config_sentence_transformers.json
Wrote 2345 bytes to dump/checkpoints/113500/README.md
Wrote 2345 bytes to dump/checkpoints/113000/README.md
Wrote 52 bytes to dump/checkpoints/112500/sentence_bert_config.json
Wrote 2345 bytes to dump/checkpoints/112500/README.md
Wrote 190 bytes to dump/checkpoints/112500/1_Pooling/config.json
Wrote 231508 bytes to dump/checkpoints/112000/vocab.txt
Wrote 1242 bytes to dump/checkpoints/112000/tokenizer_config.json
Wrote 695 bytes to dump/checkpoints/112000/special_tokens_map.json
Wrote 711649 bytes to dump/checkpoints/112000/tokenizer.json
Wrote 124 bytes to dump/checkpoints/112000/config_sentence_transformers.json
Wrote 748 bytes to dump/checkpoints/112500/config.json
Wrote 748 bytes to dump/checkpoints/112000/config.json
Wrote 52 bytes to dump/checkpoints/112000/sentence_bert_config.json
Wrote 190 bytes to dump/checkpoints/112000/1_Pooling/config.json
Wrote 1020 bytes to dump/binary_classification_evaluation_pre_train_results.csv
Wrote 349 bytes to dump/checkpoints/112000/modules.json
Wrote 2345 bytes to dump/checkpoints/112000/README.md
Wrote 133462128 bytes to dump/checkpoints/113710/model.safetensors
Wrote 133462128 bytes to dump/BAAI--bge-small-en-v1.5-ft/model.safetensors
Wrote 133462128 bytes to dump/checkpoints/112000/model.safetensors
Wrote 133462128 bytes to dump/checkpoints/112500/model.safetensors
Wrote 133462128 bytes to dump/checkpoints/113000/model.safetensors
Wrote 133462128 bytes to dump/checkpoints/113500/model.safetensors
```

## Sample Output

Currently, this code evaluates the model beforehand, trains the model, and then evaluates it afterwards. (note this is an example run with very low hyperparameter choices for speed of development)

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
