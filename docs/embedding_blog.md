# Embedding English Wikipedia under 15 minutes

Modal Labs provides a seamless serverless solution for organizations grappling with scaling up workloads as their user base expands. Their technology enables rapid scaling across many GPUs, this post we're go over everything you need to embed the entire English Wikipedia in just 15 minutes using Huggingface's Text Embedding Inference service.

Modal Labs simplifies the use of open-source embedding models, while giving you complete control of how inference is handled, helping companies bypass common hurdles like rate limits while being able to control every detail.

The objective of this post is to demonstrate how to conduct large-scale batch embedding jobs on Modal, which is crucial for enhancing RAG applications where we want to eventually finetune embeddings using feedback data, if we trained a new embedding model more frequently we'll need to re-embed data just as frequently.

In future discussions, we'll delve into using Modal to gridsearch and finetuning your own embedding models on specific data sets for more tailored user experiences. For now, we're focusing on the foundational aspects of this technology!

## Why Wikipedia?

Our goal is to demonstrate how to use Modal to embedded large datasets. This will enable companies to enhance their embedding models by fine-tuning them with their own datasets and re-embedding the source data swiftly and effectively.

1. Closed source models can not be finetuned on your own data, now matter how much user feedback you get.
2. Remote APIs can be slow (rate limits, network latency, etc), and expensive (cost of tokens rather than compute time)

Hypothetically, as we serve a RAG application, we'll get citations and feedback that we can use to determine which questions and chunks/passages are most relevant to the user. We can then use this data to finetune our embedding model to capture the nuances of our user's language and re-embed the source data to reflect this. This would allow us to serve more relevant results to our users and improve the overall experience!

## Why Modal?

Model is a Infrastructure as Code (IAC) company that aims to simplify the complex process of deploying your code. By only paying for what you use, and abstracting away all the complexity of deploying and serving, Modal provides a simplified process to help you focus on what's important - the features that you want to build.

> To follow along with some of these examples, you'll need to create a Modal account at their https://modal.com. You'll get $30 out of the box and all of the features immediately to try out. Once you've done so, make sure to install the `modal` python package using a virtual environment of your choice and you can run all of the code we've provided below.

## Concepts for Modal

Before we dive into the code, let's take a look at some of the key concepts that Modal provides that will allow us to run our embedding job quickly and efficiently. In order to understand that, we'll need to look at two concepts - a Stub and a Volume.

### Stubs

A Stub is the most basic concept in Modal — it's a description of how to create a Modal application. We can declare it using just 2 lines of code.

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

Once you have this, you'll be able to access this volume from any other app you define using the `volumes` parameter in the `stub.function` keyword. In fact, what makes this so good is that a file on your volume behaves almost identically to a file located on your local file system, yet mantains all of the advantages that Modal brings.

## Embedding Wikipedia

Now that we've got a good understanding of some key concepts that Modal provides, lets load the `wikipedia` dataset in a persistent volume we've created called `embedding-wikipedia`, set up the hugging face inference server and run our distributed batch gpu job to embed the entire dataset.

### Loading the Dataset

We'll be using the Huggingface `datasets` library to download the dataset before saving it explicitly into a directory of our choice for future use. In order to do so, we'll create our first custom docker image that has the `datasets` package installed in a file called `download.py`.

> Note here that we explicitly need to commit and save new changes to our volume. If not, these changes will be discarded once the container is shut down.

```python
from modal import Image, Volume, Stub


volume = Volume.persisted("embedding-wikipedia") # Declare a persistent Volume
image = Image.debian_slim().pip_install("datasets") # Create a custom Image with the datasets package

stub = Stub(image=image) # Assign the image to our Stub
cache_dir = "/data" # Define a cache directory that we'd like to mount our volume inside our container
@stub.function(volumes={cache_dir: volume}, timeout=3000)
def download_dataset(cache=False):
    from datasets import load_dataset

    # Download and save the dataset locally
    dataset = load_dataset("wikipedia", "20220301.en", num_proc=10)
    dataset.save_to_disk(f"{cache_dir}/wikipedia")

    # Commit and save to the volume
    volume.commit()

@stub.local_entrypoint()
def main():
    download_dataset.remote()
```

You can then run this file by using the command

```bash
modal run download.py
```

### Huggingface Embedding Inference Server

For our embedding function, we'll be using the Hugging Face [Text Embedding Inference](https://github.com/huggingface/text-embeddings-inference) server. We'll walk through how to leverage caching of model weights by defining a custom modal image, manage container state through a Modal `cls` object methods and lastly, how to leverage this new container in our other functions.

#### Parameters

Let's start by defining some parameters for the `Text Embedding Inference` program. In our case, we're specifying the specific embedding model we're using and increasing the maximum batch size so that we can speed up our embedding job.

```python
MODEL_ID = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 768

LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(BATCH_SIZE * 512),
]
```

#### Defining Our Image

On the initial run, the Text Embedding Inference package will cache and download the model weights for us to use in subsequent runs.

```python
def spawn_server() -> subprocess.Popen:
    import socket

    process = subprocess.Popen(["text-embeddings-router"] + LAUNCH_FLAGS)

    # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if launcher webserving process has exited.
            # If so, a connection can never be made.
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


def download_model():
    # Wait for server to start. This downloads the model weights when not present.
    spawn_server()
```

We'll be using the recomended image for A10G GPUs for this example but if you'd like to explore other GPU models, make sure to download the correct model listed [here](https://huggingface.co/docs/text-embeddings-inference/supported_models). We also override the default entry point so that it is compatible with Modal.

```python
tei_image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, gpu=GPU_CONFIG)
    .pip_install("httpx")
)
```

By using the `run_function` property, we're able to execute the `download_model` function as part of the build step. This allows us to bake the downloaded weights into our image, bringing our startup time down to an incredible 2-4s from around 10-15s without any caching. That's an almost 80% reduction in boot time with just a single line of code.

#### Creating our Modal Class

Using a modal class enhances control over a container's lifecycle:

1. Initialize once at boot with **enter**.
2. Handle calls from other functions using @method decorators.
3. Clean up at shutdown with **exit**.

For example, we initialize a server at boot, maintaining its state for subsequent requests, optimizing initialization costs. Modal simplifies lifecycle management by requiring only two function definitions and a decorator. Additionally, configure the stub class for specific images and GPUs through stub.cls parameters.

```python
from modal import gpu

GPU_CONFIG = gpu.A10G()

@stub.cls(
    gpu=GPU_CONFIG,
    image=tei_image, # This is defined above
)
class TextEmbeddingsInference:
    def __enter__(self):
        from httpx import AsyncClient

        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=10)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.process.terminate()

    async def _embed(self, chunk_batch):
        texts = [chunk[3] for chunk in chunk_batch]
        res = await self.client.post("/embed", json={"inputs": texts})
        return np.array(res.json())

    @method()
    async def embed(self, chunks):
        """Embeds a list of texts.  id, url, title, text = chunks[0]"""

        coros = [
            self._embed(chunk_batch)
            for chunk_batch in generate_batches(chunks, batch_size=BATCH_SIZE)
        ]

        embeddings = np.concatenate(await asyncio.gather(*coros))
        return chunks, embeddings.
```

### Generating Embeddings

Let's take stock of what we've achieved so far:

- We first created a simple Modal stub.
- Then, we created a persistent volume that could store data in between our script runs and downloaded the entirety of English Wikipedia into it.
- Next, we put together our first Modal `cls` object using the Text Embedding Inference image from docker and attached a A10G GPU to the class.
- Lastly, we defined a method we could call in other stub functions using the `@method` decorator.

Now, let's see how to use the dataset that we downloaded with our container to embed all of wikipedia. We'll first write a small function to split our dataset into batches before seeing how we can get our custom Modal `cls` object to embed all of the chunks.

#### Chunking Text

We'll be using the [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) model in order to embed all of our content. This model has a maximum sequence length of 512 tokens so we can't pass in an entire chunk of text at once. Instead, we'll split it into chunks of 400 characters for simplicity using the function below, in practice you'll want to split it more intelligently, and include overlap between chunks to avoid losing information.

```python
def generate_chunks_from_dataset(xs, chunk_size: int = 400):
    for data in xs:
        id_ = data["id"]
        url = data["url"]
        title = data["title"]
        text = data["text"]
        for chunk_start in range(0, len(text), chunk_size):
            yield (
                id_,
                url,
                title,
                text[chunk_start : chunk_start + chunk_size],
            )
```

To speed data transfer batch our `generate_chunks_from_dataset` chunks into batches of 512 chunks each. This allows us to pass in a batch of 512 chunks to our Modal `cls` object to embed at once. This is a simple function that we can write to chunk our data into batches.

```python
def generate_batches(xs, batch_size=512):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
```

#### How do we apply

After creating a function to batch our dataset, we can now pass these chunks to our Modal `cls` object for embedding. We have a custom image with the `datasets` library installed to easily load our dataset from disk. Additionally, we have implemented logic to extract a subset of the dataset.

To call our custom Modal `cls` object and use the `.embed` function with our data batches, we simply use the `.map` function. Modal takes care of managing the containers, serializing and deserializing inputs, and handling the lifecycle of each container.

```python
@stub.function(
    image=Image.debian_slim().pip_install("datasets"),
    volumes={cache_dir: volume},
    timeout=5000,
)
def embed_dataset():
  dataset = load_from_disk(f"{cache_dir}/wikipedia")
  model = TextEmbeddingsInference()

  text_chunks = generate_chunks_from_dataset(dataset["train"], chunk_size=512)
  batches = generate_batches(text_chunks, batch_size=batch_size)

  # Collect the chunks and embeddings
  for batch_chunks, batch_embeddings in model.embed.map(batches, order_outputs=False):
    ...

  return
```

### How do we run this?

We define a `local_entrypoint` to call this new stub function in our `main.py` file.

```python
@stub.local_entrypoint()
def main():
    embed_dataset()
```

This allows us to run this entire script using the command

```bash
modal run main.py
```

In a production setting you might want to run this on a schedule as new data comes in, or as you get more user feedback. This allows you update data and models in production without having to worry about the underlying infrastructure. You just need to modify the `@stub.function` decorator to add in a `schedule` parameter. This can be modified to any arbitrary period that you'd like to use depending on your use case.

```python
from modal import Period

@stub.function(
    ...
    schedule=Period(days=1)
)
```

We can then deploy this using the command

```bash
modal deploy --name wikipedia-embedding main.py
```

If you'd like to change the frequency, just change the schedule parameter, re-run the command and you're good to go!

## How do we speed this up?

Now, let's explore ways to optimize our embedding function for faster processing, speeding it from 8 hours to 15 minutes.

To speed up the process, we can make two adjustments. First, we can increase the number of containers we use. Second, we can rewrite the `embed` function to take advantage of asynchronous processing within the container. Let's examine each of these modifications and evaluate the performance improvements they offer.

### Increasing the Concurrency Limit

Modal has a cap on the number of containers that processes ar allowed to spawn concurrently to run a batch job. In this case, when we use the `.map` function, we're limited to around 10 containers that we can use at any given time. This can be easily overwritten using the `concurrency_limit` parameter on our `stub.cls` object.

All we really need to do is to then crank up the value of `concurrency_limit` to 50 and we'll have 50 different containers each wtih their own A10G GPU processing batches of text to be embedded. We suggest experimenting with these parameters to see what works best for your use case.

```python
@stub.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    concurrency_limit=50, # Number of concurrent containers that can be spawned to handle the task
)
class TextEmbeddingsInference:
    # Rest Of Code below
```

### Uploading Our Dataset

If you check out the example code we provide you'll notice that we've uploaded the dataset to a public Hugging Face dataset using huggingface, we'll provide some details in the readme on how to do this. But in practice how you handle this data will depend on your use case, either you save it to a public dataset, or you can upload it to a private dataset or insert it into a database.

# Conclusion

Today's we went through a few foundational concepts that are key to taking advantage of Modal's full capabilities - namely stubs and volumes and how they can be used in tandem to run massive parallelizable jobs at scale.

Having the ability to scale unlocks new business use cases for companies that can now iterate on production models more quickly and efficiently. By shortening the feedback loop with Modal's serverless gpus, companies are then freed up to focus on experimentation and deployment.

You can check out some datasets [here](https://huggingface.co/567-labs) containing embeddings that we computed using some popular open source embedding models. We've also uploaded our code [here](#todo) where we also showcase how to upload the generated embeddings to Hugging Face.
