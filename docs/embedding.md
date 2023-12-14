# Embedding Wikipedia under 30 minutes

Embedding hundreds of gigabytes of text data can be a daunting task, especially when limited to making batch requests to a remote API. This article explores how Modal can be used to efficiently embed the huggingface Simple English Wikipedia in under 30 minutes. By mounting the data into a Modal volume and running the embedding function in parallel across multiple GPUs, we can achieve this.

This blog is the first in a series that will demonstrate how Modal can be used to execute and refine embedding models, enabling a constantly improving RAG application. We will begin by discussing the motivation behind embedding large datasets and then explore the key concepts provided by Modal for quick and efficient embedding. Finally, we will demonstrate how Modal can be utilized to embed the entire Simple English Wikipedia in under 30 minutes.

In a future post, we will explain how feedback and relevance scores from our RAG application can be used to fine-tune our embedding model and re-embed the source data accordingly.

## Why Wikipedia?

The goal is to show how we can use Modal to embed large datasets quickly and efficiently. Which will open up new opportunities for companies to iterate on their embedding models by being able to finetune embedding models on their own datasets and reembedding the source data quickly and efficiently.

1. Closed source models can not be finetuned on your own data, now matter how much user feedback you get.
2. Remote APIs can be slow (rate limits, network latency, etc), and expensive (cost of tokens rather than compute time)

Hypothetically, as we serve a RAG application, we can use user feedback to determine qhich questions and text passages are most relevant to the user. We can then use this data to finetune our embedding model to capture the nuances of our user's language and re-embed the source data to reflect this. This would allow us to serve more relevant results to our users and improve the overall experience!

## Why Modal?

Model is a Infrastructure as Code (IAC) company that aims to simplify the complex process of deploying your code. By only paying for what you need, and abstracting away all the complexity of deploying and serving, Modal provides a simplified process to help you focus on what's important - the features that you want to build.

> To follow along with some of these examples, you'll need to create a Modal account at their https://modal.com. You'll get $30 out of the box and all of the features immediately to try out. Once you've done so, make sure to install the `modal` python package using a virtual environment of your choice and you can run all of the code we've provided below.

## Concepts

Before we dive into the code, let's take a look at some of the key concepts that Modal provides that will allow us to run our embedding job quickly and efficiently. In order to understand that, we'll need to look at two concepts - a Stub and a Volume.

### Stubs

A Stub is the most basic concept in Modal. We can declare it using just 2 lines of code.

```python
import modal

stub = modal.stub()
```

Once we've declared the stub, you can then do really interesting things such as provision GPUs to run workloads, instantiate an endpoint to serve large language models at scale and even spin up hundreds of containers to process large datasets in parallel.

**It's as simple as declaring a new property on your stub.**

Stubs can even be run with any arbitrary docker image, pinned dependencies and custom dockerfiles with the custom Rust build system Modal provides. All it takes is to define the image using Modal's `Image` object, adding it to the stub declaration and you're good to go.

### Volumes

In order to load large datasets and models efficiently, we can use Modal's Volumes feature. Volumes are a way to mount data into your containers and allow you to read and write to them as if they were a local file system. This is especially useful when you're dealing with large datasets that you'd like to load into your containers quickly and efficiently.

```python
import modal

stub = modal.Stub()
stub.volume = modal.volume.new() # Declare a new volume

@stub.function(volumes={"/root/foo": stub.volume})
def function():
  // Perform tasks here
```

If you'd like to ensure that the data in the volume is permanently stored on Modal's servers and can be accessed by any of your other modal functions, all it takes is just swapping out the `.new` for the `.persisted` keyword

```python
volume = Volume.persisted("Unique Name Here")
```

Once you have this, you'll be able to access this volume from any other app you define using the `volumes` parameter in the `stub.function` keyword. In fact, what makes this so good is that a file on your volume behaves almost identically to a file located on your local file system, yet mantains all of the advantages that Modal brings.

In some of our initial test runs, we were able to load in the entirety of Simple English Wikipedia, all 22 GB of it, into a container in just under 5 seconds and start executing jobs.

## Embedding Wikipedia

Now that we've got a good understanding of some key concepts that Modal provides, lets load the `wikipedia` dataset in a persistent volume we've created called `embedding-wikipedia`, set up the hugging face inference server and run our distributed batch gpu job to embed the entire dataset.

<TODO : Add in a simplified script to run embeddings without the sample portion>

## Scaling Out

Now that we've seen how to run a simple batch job using Modal, let's consider how we might modify our embedding function to bring our current timing of ~4 hours down.

Well, turns out we have two handy properties that we can adjust

- `concurrency_limit` : This is the number of containers that Modal is allowed to spawn concurrently to run a batch job
- `allow_concurrent_inputs` : This is the number of inputs a container can handle at any given time

All we really need to do then is to crank up the values of these two properties as seen below and we'll have 50 different containers each with their own A10g GPU processing 40 batches of inputs each at any given time.

```python
@stub.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    concurrency_limit=50, # Number of concurrent containers that can be spawned to handle the task
    allow_concurrent_inputs=40, # Number of inputs each container can process and fetch at any given time
)
class TextEmbeddingsInference:
    //Rest Of Code
```

With these two new lines of code, we can cut down the time taken by almost 90%, from 4 hours to 30 minutes without any need for any complex optimisations or specialised code, just by using Modal's built in features.

# Conclusion

Today's we went through a few foundational concepts that are key to taking advantage of Modal's full capabilities - namely stubs and volumes and how they can be used in tandem to run massive parallelizable jobs at scale.

Having the ability to scale unlocks new business use cases for companies that can now iterate on production models more quickly and efficiently. By shortening the feedback loop with Modal's serverless gpus, companies are then freed up to focus on experimentation and deployment.

We've uploaded our code [here](#todo). You can also check out some datasets [here](https://huggingface.co/567-labs) containing embeddings that we computed using some popular open source embedding models.
