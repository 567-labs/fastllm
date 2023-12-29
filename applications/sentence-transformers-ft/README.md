# README

## Summary

This directory explores finetuning sentence transformers with Modal.

It covers three main fine-tuning topics:

* `finetune_OnlineContrastiveLoss.py`: this file contains code to finetune a SentenceTransformer model locally. It uses OnlineContrastiveLoss to finetune the quora pairs dataset on the task of duplicate question classification.
* `modal_main.py`: this file runs finetuning code on Modal
* `modal_optuna.py`: this file runs finetuning cod with grid search using Optuna on Modal

It also covers parallel evaluation of base models using Modal with `eval.py`.

Draft blog posts can be found in `/docs/finetune-embedding.md` and `/docs/finetune-embedding-pt2.md`.

## Example

An example model that has been fine-tuned with this repo can be found here: <https://huggingface.co/krunchykat/bge-base-en-v1.5-ft-quora>

`krunchykat/bge-base-en-v1.5-ft-quora` was trained with the following hyperparameters using Modal:

        * model_id: str = "BAAI/bge-small-en-v1.5"
        * epochs: int = 10
        * activation_function=nn.Tanh(),
        * scheduler: str = "warmuplinear",
        * dense_out_features: Optional[int] = None,
        * batch_size: int = 32,

Metrics before training

        ```csv
        epoch,steps,cossim_accuracy,cossim_accuracy_threshold,cossim_f1,cossim_precision,cossim_recall,cossim_f1_threshold,cossim_ap,manhattan_accuracy,manhattan_accuracy_threshold,manhattan_f1,manhattan_precision,manhattan_recall,manhattan_f1_threshold,manhattan_ap,euclidean_accuracy,euclidean_accuracy_threshold,euclidean_f1,euclidean_precision,euclidean_recall,euclidean_f1_threshold,euclidean_ap,dot_accuracy,dot_accuracy_threshold,dot_f1,dot_precision,dot_recall,dot_f1_threshold,dot_ap
        -1,-1,0.7832743822503648,0.8544833064079285,0.7340090477008532,0.63631850675139,0.8671357055878771,0.8132455348968506,0.7718542072654618,0.7833733211308714,11.693254470825195,0.7333426926856661,0.6268420294724716,0.8834393180895684,13.734819412231445,0.7718763407421714,0.7832743822503648,0.5394750833511353,0.7340090477008532,0.63631850675139,0.8671357055878771,0.6111537218093872,0.7719264054777693,0.7832743822503648,0.8544832468032837,0.7340090477008532,0.63631850675139,0.8671357055878771,0.8132455348968506,0.7718060771210138
        ```

* cossim_ap = 0.7718542072654618

Metrics after training

        ```csv
        epoch,steps,cossim_accuracy,cossim_accuracy_threshold,cossim_f1,cossim_precision,cossim_recall,cossim_f1_threshold,cossim_ap,manhattan_accuracy,manhattan_accuracy_threshold,manhattan_f1,manhattan_precision,manhattan_recall,manhattan_f1_threshold,manhattan_ap,euclidean_accuracy,euclidean_accuracy_threshold,euclidean_f1,euclidean_precision,euclidean_recall,euclidean_f1_threshold,euclidean_ap,dot_accuracy,dot_accuracy_threshold,dot_f1,dot_precision,dot_recall,dot_f1_threshold,dot_ap
        -1,-1,0.9099903534591506,0.8681730031967163,0.8798801732278336,0.8482013936844749,0.9140170477607902,0.852232813835144,0.9248234406964589,0.9101387617799105,11.39983081817627,0.8795641740709514,0.8446686596910812,0.9174671898254634,12.09014892578125,0.9248780629159912,0.9099903534591506,0.5134725570678711,0.8798801732278336,0.8482013936844749,0.9140170477607902,0.5436307787895203,0.9249197879998878,0.9099903534591506,0.8681729435920715,0.8798801732278336,0.8482013936844749,0.9140170477607902,0.852232813835144,0.9247845722254051
        ```

* cossim_ap = 0.9248234406964589

You can use this model like any other HuggingFace compatible model with the SentenceTransformers library.

        ```python
        from sentence_transfomers import SentenceTransformer
        model = SentenceTransformer("krunchykat/bge-base-en-v1.5-ft-quora")
        emb = model.encode("hello world")
        ```

We benchmark this fine-tuned model to other base sentence embedding models in [eval_metrics.csv](./examples/eval_metrics.csv).
Note that we only use the test set to evaluate in the [eval script](./eval.py)

## Instructions

### Run locally 

1. Make sure the hyperparameters you want are set correctly. Then initiate training with

        ```bash
        python finetune_OnlineContrastiveLoss.py
        ```

2. The model, evals, and checkpoints will be automatically saved to `./output` path in your local directory.

### Run with Modal 

1. Make sure the hyperparameters you want are set correctly in [modal_main.py](./modal_main.py). Then initiate training with

        ```bash
        modal run modal_main.py
        ```

2. After the modal app is initiated, you should see the app running in the Modal dashboard as well as a new storage volume where your finetuned model/metrics will be stored. This volume will be named `sentence-transformers-ft-{UNIX_TIMESTAMP}` Once training finishes, you can download the volume to your local directory by running. This saves your model, checkpoints, and evals to your local computer.

        ```bash
        mkdir output
        modal volume get sentence-transformers-ft-{UNIX_TIMESTAMP} /"**" ./output --force
        ```

        //Remember, you can find the {UNIX_TIMESTAMP} in the Modal dashboard "storage tab"

Example pre-train metrics can be found [here](./examples/binary_classification_evaluation_pre_train_results.csv) and example post-train metrics can be found [here](./examples/binary_classification_evaluation_post_train_results.csv)

3. You're finetuned model will be stored in `./output/{BASE_MODEL}-ft`. You can upload it to huggingface by following these procedures

    a. Create a Huggingface account, install the [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/guides/cli) and login to the cli with your account.

    b. Create a new model using the [Huggingface dashboard](https://huggingface.co/new). Remember the model name for the next command.

    c. Run the following command to upload your model. Note the `MODEL_DIRECTORY` is in the format `{BASE_MODEL}-ft`

        `huggingface-cli upload <huggingface_username>/<model_name> ./output/<MODEL_DIRECTORY> 

    d. You can now use your Huggingface model like any other Huggingface compatible sentence transformer. This is an example model I uploaded: <https://huggingface.co/krunchykat/bge-base-en-v1.5-ft-quora>

### Run with optuna 

1. Make sure the hyperparameters you want are set correctly in [modal_optuna.py](./modal_optuna.py). Then initiate training with

        ```bash
        modal run modal_optuna.py
        ```
2. After training finishes, you can view all trials' metrics by downloading the trials csv data using Modal Volumes cli

        ```bash
        modal volume get sentence-transformers-ft-optuna-{UNIX_TIMESTAMP} ./trials.csv . 
        ```

        //Remember, you can find the {UNIX_TIMESTAMP} in the Modal dashboard "storage tab"

This will return something like [trials.csv](./examples/trials.csv)

3. List the specific finetuned model, metrics, and checkpoints for a specific trial using

        ```bash
        modal volume ls sentence-transformers-ft-optuna-{UNIX_TIMESTAMP} trial-{TRIAL_NUMBER}
        ```

4. Download a trial's finetuned model, metrics, and checkpoints using this command.

        ```bash
        modal volume get sentence-transformers-ft-optuna-{UNIX_TIMESTAMP} /trial-{TRIAL_NUMBER}/"**" ./output
        ```

5. Follow the steps in [#run-with-modal](#run-with-modal) to upload to huggingface

### Run Evals on multiple embedding models

1. Specify the embedding models you want to evaluate in [eval.py](./eval.py).

2. Run with Modal

        ```python
        modal run eval.py
        ```

3. Metrics will be saved locally in `eval_metrics.csv`. An example can be found [here](./examples/eval_metrics.csv)