# Embedding All of Wikipedia under 30 minutes

Embedding hundreds of gigabytes of text data can be a daunting task, especially when limited to making batch requests to a remote API. This article explores how Modal can be used to efficiently embed the huggingface Simple English Wikipedia in under 30 minutes. By mounting the data into a Modal volume and running the embedding function in parallel across multiple GPUs, we can achieve this.

This blog is the first in a series that will demonstrate how Modal can be used to execute and refine embedding models, enabling a constantly improving RAG application. We will begin by discussing the motivation behind embedding large datasets and then explore the key concepts provided by Modal for quick and efficient embedding. Finally, we will demonstrate how Modal can be utilized to embed the entire Simple English Wikipedia in under 30 minutes.

In a future post, we will explain how feedback and relevance scores from our RAG application can be used to fine-tune our embedding model and re-embed the source data accordingly.

## Why Wikipedia?

The goal is to show how we can use Modal to embed large datasets quickly and efficiently. Which will open up new opportunities for companies to iterate on their embedding models by being able to finetune embedding models on their own datasets and reembedding the source data quickly and efficiently.

1. Closed source models can not be finetuned on your own data, now matter how much user feedback you get.
2. Remote APIs can be slow (rate limits, network latency, etc), and expensive (cost of tokens rather than compute time)

Hypothetically, as we serve a RAG application, we can use user feedback to determine which questions and text passages are most relevant to the user. We can then use this data to finetune our embedding model to capture the nuances of our user's language and re-embed the source data to reflect this. This would allow us to serve more relevant results to our users and improve the overall experience!

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

## Setup

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

### Text Embedding Inference

In order to embed our function, we'll be using the Hugging Face [Text Embedding Inference](https://github.com/huggingface/text-embeddings-inference) package. We'll walk you through how to leverage caching of model weights by defining a custom modal image, manage container state through a Modal `cls` object methods and lastly, how to leverage this new container in our other functions.

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

A modal class allows us to have more fine-grained control over the behaviour of our container. This allows us to separate different portions of the container life cycle

- What to do just once when the container boots up using the  `__enter__` 
- What should we do when the container is called by other functions by defining  `@method` calls
- What to do once the container is shut down using the `__exit__` 

More specifically, in our case, we spawn a server once when the container boots up. This state is then preserved in preparation for future requests that other functions might make so that we only incur the cost of initialising a server once. 

All the guesswork of managing the life cycle is taken out of the equation for you with Modal by just defining two functions and using a single decorator. Not only so, we can configure our object to run with a specific image and an attached GPU by modifying the `stub.cls` parameters

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

Let's take stock of what we've achieved so far. We first created a simple Modal stub. Then, we created a persistent volume that could store data in between our script runs and downloaded the entirety of English Wikipedia into it. Next, we put together our first Modal`cls` object using the Text Embedding Inference image from docker and attached a A10G GPU to the class. Lastly, we defined a method we could call in other stub functions using the `@method` decorator.

Now, let's see how to use the dataset that we downloaded with our container to embed all of wikipedia. We'll first write a small function to split our dataset into batches before seeing how we can get our custom Modal`cls` object to embed all of the chunks.

#### Chunking Text

We'll be using the [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) model in order to embed all of our content. This model has a maximum sequence length of 512 tokens so we can't pass in an entire chunk of text at once. Instead, we'll split it into chunks of 400 characters for simplicity using the function below. 


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

To speed things up, we take the list of chunks that our `generate_chunks_from_dataset` has created and convert them into a list of batches that have 512 elements each. This means that in our new list, each element will be a list of at most 512 chunks.

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

#### Calling our Model

Now that we've written a function to chunk our dataset into batches, let's see how we can pass these chunks to our Modal `cls` object to embed. We also defined a custom image that has the `datasets` library installed so that we can easily load our dataset from disk. We also define some simple logic to get a subset of the entire dataset. 

Note that for us to be able to call our custom Modal `cls` object and call the `.embed` function with our batches of data, all it took was a `.map` function. Modal automatically handles the orchestration of the different containers, the serialization and deserialization of inputs between the different functions and the individual container lifecycle.

Trying to implement this on our own would have been a serious challenge - you would be constrained by the physical number of GPUs you could afford, then configure these images to work on each GPU manually before writing the code to pass data in between each container while trying to optimize throughput and utilisation of each GPU. **This is a non-trivial amount of work to accomplish**. 

```python
@stub.function(
    image=Image.debian_slim().pip_install("datasets"),
    volumes={cache_dir: volume},
    timeout=5000,
)
def embed_dataset():
	dataset = load_from_disk(f"{cache_dir}/wikipedia")
  model = TextEmbeddingsInference()
  
  ttl_size = 19560538957 # This is the size of the dataset
  sample_size = int(ttl_size * 0.01) # We work with 1% of the dataset 
  subset = dataset["train"].select(range(sample_size))
  
	text_chunks = generate_chunks_from_dataset(subset, chunk_size=512)
  batches = generate_batches(text_chunks, batch_size=batch_size)
  acc_chunks = []
  embeddings = []
  for batch_chunks, batch_embeddings in model.embed.map(batches, order_outputs=False):
      acc_chunks.extend(batch_chunks)
      embeddings.extend(batch_embeddings)
 
	return
```

### Uploading Our Dataset

Now that we've generated our new set of embeddings, we can upload it onto Hugging Face to share our new dataset. In order to do so, you'll need to create a Modal [secret](https://modal.com/docs/guide/secrets) for Hugging Face. This allows us to provide the environment variables that Hugging Face expects in order to interact with its hosting service. Let's start by first converting our `batch_embeddings` and `batch_chunks` arrays into a Hugging Face `Datasets` object. This helps to simplify a lot of the integration with hugging face.

```python
import pyarrow as pa

table = pa.Table.from_arrays(
  [
      pa.array([chunk[0] for chunk in acc_chunks]),  # id
      pa.array([chunk[1] for chunk in acc_chunks]),  # url
      pa.array([chunk[2] for chunk in acc_chunks]),  # title
      pa.array([chunk[3] for chunk in acc_chunks]),  # text
      pa.array(embeddings),
  ],
  names=["id", "url", "title", "text", "embedding"],
) 

dataset = Dataset(table)
```

Since our dataset is rather large, we can implement an intermediate checkpoint by saving our dataset into a Modal volume before proceeding with the upload. 

```python
checkpoint_volume = Volume.persisted("checkpoint")
checkpoint_dir = "/checkpoint"
hf_dataset_name = "567-labs/wikipedia-embeddings"

@stub.function(
    image=Image.debian_slim().pip_install("datasets"),
    volumes={cache_dir: volume,checkpoint_dir:checkpoint_volume},
    timeout=86400, # Increase timeout to 24 hours
    secret=Secret.from_name("huggingface-credentials"),
)
def embed_dataset():
	# ... Rest of Code
  from datasets import Dataset
	dataset = Dataset(table)
	checkpoint_path = f"{checkpoint_dir}/wikipedia-embeddings"
	dataset.save_to_disk(checkpoint_path)
	checkpoint_volume.commit()
	
	dataset.push_to_hub(hf_dataset_name)
	
```

With this, we'll now be able to automatically embed and upload our embeddings onto hugging face when we run our script. What makes this even better is that we've got an intermediate checkpoint by simply adding in a few lines of code that allows us to resume uploads in the event that we get any unexpected errors.

Based on our initial runs, this takes around 1.5 hrs to complete.

### Running the Code

We define a `local_entrypoint` to call this new stub function in our `main.py` file. 

```python
@stub.local_entrypoint()
def main():
	embed_dataset()
```

This allows us to run this entire script using the command

```python
modal run main.py
```

Alternatively, if you'd like to run this as a cron job daily, you just need to modify the `@stub.function` decorator to add in a `schedule` parameter. This can be modified to any arbitrary period that you'd like to use depending on your use case.

```python
from modal import Period

@stub.function(
    image=Image.debian_slim().pip_install("datasets"),
    volumes={cache_dir: volume},
    timeout=5000,
    schedule=Period(days=1)
)
```

We can then deploy this using the command

```bash
modal deploy --name wikipedia-embedding main.py
```

If you'd like to change the frequency, just change the schedule parameter, re-run the command and you're good to go. This makes it perfect for companies exploring to finetune their embeddings on a regular basis without having to worry about the underlying infrastructure. 

## Scaling Out

Now that we've seen how to run a simple batch job using Modal, let's consider how we might modify our embedding function to speed up the time taken. Currently if we run our script above, embedding all of wikipedia will take around 8 hours with a batch size of 512. 

To do so, there are two things that we can do - increase the number of containers we're using and rewriting our `embed` function to take advantage of asynchronous processing within the container. Let's tackle them one by one and see what performance benefits we can get.

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

### Asynchronous Processing

By handling multiple requests concurrently instead of sequentially, we can significantly improve the performance of our `embed` function, allowing us to reduce the time taken to perform the embeddings. This is done by using a `_embed` function to process each batch in parallel.

```python
async def _embed(self, chunk_batch):
    texts = [chunk[3] for chunk in chunk_batch]
    res = await self.client.post("/embed", json={"inputs": texts})
    return np.array(res.json())
```

Each time we call this embed function, it sends the chunked batch to our Text Embedding Inference server. Once it gets a response, it will then start converting it to a json response. This means that while we're waiting for our server to process a batch of chunks, other tasks such as converting the returned embeddings to a numpy array can continue executing. 

This helps us to reduce the amount of time that our program would spend idling waiting for the Text Embedding Inference server to finish processing all of the embeddings.  Since we need to have all of the embeddings on hand before, we're forced to use `asyncio.gather` here to wait for our server to finish computing all of the embeddings.

```
@method()
async def embed(self, chunks):
    """Embeds a list of texts.  id, url, title, text = chunks[0]"""

    coros = [
        self._embed(chunk_batch)
        for chunk_batch in generate_batches(chunks, batch_size=BATCH_SIZE)
    ]

    embeddings = np.concatenate(await asyncio.gather(*coros))
    return chunks, embeddings
```

By doing so, we can increase the batch size that each container can process by a lot - in our case, we can now process almost 25600 chunks per container at any given time, resulting in a **33x increase in the capacity of each container** from our original batch size of 768. 

With these two optimisations, we can now process the entirety of Wikipedia in just under 30 hours, resulting in almost 80% decrease in time taken to process the entire job.

## HF_Transfer

We can speed up the file upload speed by using [hf_transfer](https://github.com/huggingface/hf_transfer). This is a tool which allows us to circumvent the rate limits imposed on upload speeds by python. More importantly, it allows us to take advantage of the high network speeds that Modal provides, potentially reaching upload speeds of up to 500mb/s. 

In order to support the `hf_transfer` package, we need to first install it by updating our custom image definition and setting the `HF_HUB_ENABLE_HF_TRANSFER` environment variable to be `1` as seen below

```python

@stub.function(
    image=Image.debian_slim().pip_install(
        "datasets", "pyarrow", "tqdm", "hf_transfer", "huggingface_hub"
    ),
    volumes={cache_dir: volume,checkpoint_volume_path:checkpoint_volume},
    timeout=86400,
    secret=Secret.from_name("huggingface-credentials"),
)
def embed_dataset():
  # rest of code
  os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

Once we've done so, we then need to modify our upload logic so that we utilise the `upload_from_file` function instead. This gives us the option of using the new experimental feature `multi_commits`, which splits up the file uploads into numerous individual files before restoring it server side.

```
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
api.create_repo(repo_id=hf_dataset_name, private=False, repo_type="dataset",exist_ok=True)
api.upload_folder(
    folder_path=path_parent_folder,
    repo_id = hf_dataset_name,
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True
)
```

With this new optimisation, we can bring down the time taken for file uploads from our original 1.5hrs to around 30 minutes, shaving off almost 70% of the original time taken to upload the data. 

# Conclusion

Today's we went through a few foundational concepts that are key to taking advantage of Modal's full capabilities - namely stubs and volumes and how they can be used in tandem to run massive parallelizable jobs at scale. 

Having the ability to scale unlocks new business use cases for companies that can now iterate on production models more quickly and efficiently. By shortening the feedback loop with Modal's serverless gpus, companies are then freed up to focus on experimentation and deployment.

You can check out some datasets [here](https://huggingface.co/567-labs) containing embeddings that we computed using some popular open source embedding models. We've also uploaded our code [here](#todo) where we also showcase how to upload the generated embeddings to Hugging Face.