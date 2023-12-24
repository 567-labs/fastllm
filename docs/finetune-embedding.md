# Boost your RAG: Fine-Tuning Text Embedding Models on Modal

Simply using a text embedding model + Vector DB is **not** a catch-all technique for building a performant retrieval system (i.e. for RAG). Rather, each task that involves using an embedding model is a unique one, and should be treated as so.

Having an embedding model that is fine-tuned for your specific task can greatly improve performance. There are a few ways of going about this: picking a performant embedding model, having a robust dataset to train your specific task on, and choosing the right way to train your embedding model. In this article, we will discuss how to approach all of these and specifically how to fine-tune an open-source sentence embedding model using [Modal](https://modal.com/).

This article is the second in a series demonstrating how Modal can be effectively used with open-source embedding models, the first can be [found here](https://www.example.com). We will discuss three main topics: preparing your dataset for fine-tuning, fine-tuning your embedding model on Modal with [SentenceTransformers](https://www.sbert.net/index.html), and making the most of fine-tuning by using [Optuna](https://optuna.org/) (hyperparameter optimization framework) on Modal.

## Fine-Tuning Basics

To begin fine-tuning a sentence transformer, we need to start by identifying the actual task we are trying to fine-tune for. Remember, every task is unique, so they require a unique dataset format, loss, and evaluator. We will be using the [SentenceTransformers](https://www.sbert.net/) framework to easily setup these requirements for fine-tuning.

For the purposes of this demo, we will be choosing the specific task of **Duplicate Questions Classification**, i.e. given two questions, identify whether they are semantically duplicate. However, there are other tasks that can be optimized for including information retrieval, duplicate statement mining, and more. Details for this can be found [here](https://www.sbert.net/examples/training/quora_duplicate_questions/README.html).

**TODO: insert table about different tasks here**

### Choosing a Base Model

Since we are fine-tuning an open-source embedding model for the task of duplicate questions

something about MTEB as a reference

### Dataset Format

### Choosing a Loss Function

### Choosing an Evaluator

## Dataset Format

* choose a base model, loss, and evaluator

## Fine-Tuning on Modal

* create a stub, set the GPU and stuff

## Fine-Tuning using Hyperparameter Optimization on Modal + Optuna

* NFS, use as the backend for Optuna
* Objective function, this is what we optimize
