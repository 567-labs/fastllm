# Concurrent Reads faster than your local machine

In this short Article, we'll walk through the new Volume feature that Modal just released. We cover a few things - from a conceptual mental model to help you get up and running to some sample implementations to show you how they work in the wild.

In case you're unfamiliar with Modal, they're a Infrastructure as Code (IAC) company that aims to simplify the complex process of deploying your code. By only paying for what you need, and abstracting away all the complexity of deploying and serving, Modal provides a simplified process to help you focus on what's important - the features that you want to build.

> To follow along with some of these examples, you'll need to create a Modal account at their https://modal.com. You'll get $30 out of the box and all of the features immediately to try out. Once you've done so, make sure to install the `modal` python package using a virtual environment of your choice and you can run all of the code we've provided below.

To do so, they use a concept called a `Stub` which you can instantiate using the `modal.stub()` definition. 

```python
import modal

stub = modal.stub()
```

With this `stub`, you can then do really interesting things such as provision GPUs to run workloads, instantiate an endpoint to serve large language models at scale and even spin up hundreds of containers to process large datasets in parallel. It's as simple as declaring a new property on your stub.

But, once we start using these containers and systems, we then face a big problem - if we've got huge models, large datasets or files to work with, how can we get the data into these containers efficiently and quickly? Remember, a 70B model is going to be almost 130GB in size, if you're looking to download that using a simple curl command, you're going to be waiting for a very long time.

That's where the new Volume feature comes in.

## What is a Volume?

Modal recently released their new Volumes feature which promises fast concurrent reads with write-once, read-many I/O workloads. This is ideal for jobs where we might need to run large batch jobs or deploy models which would otherwise take eons to download. It's suprisingly simple to set-up, all you need to create a new volume is the following 5 lines of code

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
volume = Volume.persisted(<Unique Name Here>)
```

Once you have this, you'll be able to access this volume from any other app you define using the `volumes` parameter in the `stub.function` keyword. In fact, what makes this so good is that a file on your volume behaves almost identically to a file located on your local file system. Let's see this in action to get an idea for what this means

## Downloading English Wikipedia

As always, things are easier with examples, so let's try to see how fast a Modal volume can be. To do so, we'll use the [Wikipedia Dataset](https://huggingface.co/datasets/wikipedia) from Hugging Face, we'll be specifically using the English subset which is around 22GB in size.

### Using a local script

Let's write a short function to help download and load in this dataset using the HuggingFace `datasets` library. 

```python
from datasets import load_dataset, load_from_disk
import os

data_dir = "data"
cache_dir = os.path.join(os.getcwd(), data_dir)
print(cache_dir)
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Step 1: Download the 'wikipedia' dataset and cache it in a specified directory
dataset = load_dataset("wikipedia", "20220301.en", num_proc=10)

# Print a message to indicate that the dataset has been downloaded and cached
print("Dataset downloaded and cached.")

# Step 2: Save the dataset to disk
# The dataset is already saved to disk in the previous step when specifying the 'cache_dir' parameter
# Step 2: Save the dataset to disk
dataset.save_to_disk(cache_dir.__str__())

# Step 3: Load the dataset from the specified directory
loaded_dataset = load_from_disk(cache_dir.__str__())

# Step 3: Load the dataset from the specified directory
loaded_dataset = load_from_disk(cache_dir.__str__())

# Print a message to indicate that the dataset has been loaded
print("Dataset loaded from disk.")

# Print the loaded dataset
print(loaded_dataset)
```

Using this short function, we first create a new cache directory called data in our current working directory. We then proceed to download the dataset from hugging face before saving it explicitly to our cache directory. Once that's done, we then load it from the cache directory. 

```bash
Downloading builder script: 100%|███████████████████████████████████████████████| 35.9k/35.9k [00:00<00:00, 17.0MB/s]
Downloading metadata: 100%|█████████████████████████████████████████████████████| 30.4k/30.4k [00:00<00:00, 15.8MB/s]
Downloading readme: 100%|███████████████████████████████████████████████████████| 16.3k/16.3k [00:00<00:00, 17.2MB/s]
Downloading: 100%|██████████████████████████████████████████████████████████████| 15.3k/15.3k [00:00<00:00, 17.4MB/s]
Downloading:  42%|██████████████████████████████████████████████████████████████| 20.3G/20.3G [00:00<00:00, 17.4MB/s]
```

How long does this step take?

On my local machine with around 20MB/s download speed from Hugging Face, it takes me around 20 minutes to download the whole 20GB of data. However, once that's loaded, it takes around 40 seconds to load the full 20GB into memory using the datasets library. Well, then how about working with a Modal volume instead?

### First steps with Modal

Turns out, since Volumes work almost identically to a local file system, we can just copy and paste our code with some small modifications

```python
# Step 1 : Define a custom path we want to store data in
cache_dir = "/data" 

# Step 2 : Define the Stub that we'll be using - in this case, we need a persisted volume and an image that contains the 
# Hugging
volume = Volume.persisted("embedding-wikipedia")
image = Image.debian_slim().pip_install("datasets")
stub = Stub(image=image)

# Step 3: Define a function to download the dataset. Note here we set a high timeout of 50 minutes
@stub.function(volumes={cache_dir: volume}, timeout=3000)
def download_dataset():
    # Redownload the dataset
    from datasets import load_dataset
    import time

    start = time.time()
    # We set a high `num_proc` value so that we can 
    dataset = load_dataset("wikipedia", "20220301.en", num_proc=10)
    end = time.time()
    print(f"Download complete - downloaded files in {end-start}s")
    
    # We explicitly save the dataset to the volume under a subfolder called Wikipedia
    dataset.save_to_disk(f"{cache_dir}/wikipedia")
    volume.commit()

@stub.function(volumes={cache_dir: volume})
def check_dataset_exists():
    import time

    volume.reload()
    from datasets import load_from_disk

    print("Loading dataset from disk...")
    start = time.time()
    dataset = load_from_disk(f"{cache_dir}/wikipedia")
    end = time.time()
    print(f"Took {end-start} to load dataset from disk")
    print(dataset)  # Print out a quick summary of the dataset


@stub.local_entrypoint()
def main():
    import time
    download_dataset.remote(True)
    start = time.time()
    check_dataset_exists.remote()
    end = time.time()
    print(
        f"Took {end-start}s to initialise a container with a dataset and execute code"
    )

```

Let's now try running this function to see how fast we can download the whole dataset and read it in a second function

```bash
Downloading metadata: 100%|███████████████████████████████████████████████████████| 30.4k/30.4k [00:00<00:00, 53.5MB/s]
Downloading readme: 100%|█████████████████████████████████████████████████████████| 16.3k/16.3k [00:00<00:00, 65.1MB/s]
Downloading: 100%|████████████████████████████████████████████████████████████████| 15.3k/15.3k [00:00<00:00, 43.0MB/s]
Downloading:  42%|████████████████████████████████████████████████████████████████| 20.3G/20.3G [00:00<00:00, 74.4MB/s]
// Some other logs
Loading dataset from disk...
Took 39.7626953125 to load dataset from disk
DatasetDict({
    train: Dataset({
        features: ['id', 'url', 'title', 'text'],
        num_rows: 6458670
    })
})
Dataset loaded.
Took 44.572978019714355s to initialise a container with a dataset and execute code
```

We can see that immediately off the bat, we're getting almost a 4x increase in download speed by running the download step with Modal. But what about the time needed to initialise and read in the dataset in our `check_dataset_exists` function? 

**Well, turns out it took a good 5 seconds to initialise a container, allocate the volume and stream in almost 20GB of data**. To put it into context, we took 20 minutes to download the dataset locally, around 6 minutes to download it with Modal and 5 seconds to load it from a Modal volume. **That's an almost 240x speedup by using a Modal volume as compared to loading the dataset from Hugging Face with just a few lines of code**.

## Embedding All Of English Wikipedia

Now that we've seen some of the benefits of using Modal, let's build on our previous example to see how we can embed all of english wikipedia with the `bge-base-en-v1.5` model in just under 5 hours under the free tier. 

On a high level, we'll be creating a modal app that will

1. Load in the original data from wikipedia we download
2. Split and chunk it into text that is within the chunk size supported by the `bge` model
3. Automatically distribute and partition the chunks across a number of  containers with A10Gs attached to embed using the Hugging Face `Text Embedding Inference` Library 

We'll build this step by step and by the end of it all, you'll have a working embedding script that can take in any arbitrary dataset and run an embedding job on it. Note that in our provided code, we work on a tiny subset of the code ( ~ 1-2% ) so that it runs quick and fast.

To start, let's begin by creating a new file - `main.py` which will hold all of our Modal code.

### Chunking our Dataset

Modal allows us to run scripts using a local entrypoint, much like how we would run a script. In our case, we can do so by initialising a script as seen below.

```python
from modal import Volume, Stub, Image

# We start by defining some constants
dataset_name = "wikipedia"
volume = Volume.persisted("embedding-wikipedia")
cache_dir = "/data"

# Step 1 : Define the Stub
stub = Stub("embeddings")

# Step 1.5 : Define helper functions
def generate_chunks_from_dataset(xs, chunk_size=400):
    for data in xs:
        text = data["text"]
        for chunk_start in range(0, len(text), chunk_size):
            yield text[chunk_start : chunk_start + chunk_size]

def generate_batches(xs, batch_size=32):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Step 2 : Define an image which has the datasets library that we can use to load our dataset and a defined volume
@stub.function(
    image=Image.debian_slim().pip_install("datasets"),
    volumes={cache_dir: volume},
    timeout=5000,
)
def embed_dataset(down_scale=0.02,batch_size=32):
    from datasets import load_from_disk
    import time

    start = time.perf_counter()
    # Load the dataset as a Hugging Face dataset
    print("Loading dataset from disk...")
    dataset = load_from_disk(f"{cache_dir}/wikipedia")
    print(f"Dataset loaded in {time.perf_counter()-start:.2f} seconds")

    # Extract the total size of the dataset
    ttl_size = len(dataset["train"])

    # Counting all characters in the dataset
    dataset_chars = 19560538957  # sum(map(len, dataset["train"]["text"]))
    print(f"Total dataset characters: {dataset_chars}")

    sample_size = int(ttl_size * down_scale)
    print(f"Calculated dataset size of {ttl_size} and sample size of {sample_size}")

    # Iterate over the first 5% of the dataset's rows
    subset = dataset["train"].select(range(sample_size))

    print(f"Working with {sample_size} rows")
    text_chunks = generate_chunks_from_dataset(subset, chunk_size=400)
    batches = generate_batches(
        text_chunks, batch_size=batch_size
    )  # 3

    return


# Step 3 : Define a Local Entrypoint. This will be called when you run the file.
@stub.local_entrypoint()
def main():
    embed_dataset.remote(down_scale=0.05,batch_size=32)
```

We can then run this small chunk of code by using the command 

```
modal run main.py
```

This in turn generates the output of 

```
✓ Initialized. View run at https://modal.com/apps/ap-XhMcTEI8cf6ZNTcGUdve5a
✓ Created objects.
├── 🔨 Created mount /Users/admin/Documents/567/fastllm/applications/wikipedia/a.py
└── 🔨 Created embed_dataset.
Loading dataset from disk...
Dataset loaded in 36.35 seconds
Total dataset characters: 19560538957
Calculated dataset size of 6458670 and sample size of 322933
Working with 322933 rows
Stopping app - local entrypoint completed.
```

With this, we have a simple file that we can call from our local machine to load a subset of our wikipedia dataset and generate 400 character chunks from it. We then convert these chunks into batches which contain 32 chunks each.

### Embedding our Chunks

Now that we're written up the first part of our script, we now need to create a stub function which will be able to take in some text as input and return a list of embeddings using our chosen `bge-base-en-v1.5` model. In order to do so, we'll be using the Hugging Face Text Embedding Inference docker image. Since we're using the A10s, we'll be using the ` "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0" ` Docker Image.

Modal supports docker images out of the box, so to create a new stub function that has an attached A10 GPU and is based off the Docker Image that we want, all we need to do is to declare it as parameters on an `Image` object. 

```python
from modal import gpu, Image,
# Step 1: Define global config variables
GPU_CONFIG = gpu.A10G()
MODEL_ID = "BAAI/bge-base-en-v1.5" # Our embedding model of choice
BATCH_SIZE = 256 * 2
DOCKER_IMAGE = (
    "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0"
)
N_GPU = 10 # The number of concurrent images that we'd like to use at the same time

# Step 2 : Define the Config to run our Text Embedding Inference server
LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--max-client-batch-size",
    str(BATCH_SIZE),
    "--max-batch-tokens",
    str(16384 * 3),
]

# Step 3 : Define a function to initialise our server when our container is loaded
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

# Step 4 : Define an image which uses our desired image, has allocated GPUs and spins up a server on load
tei_image = (
    Image.from_registry(
        "ghcr.io/huggingface/text-embeddings-inference:86-0.4.0",
        add_python="3.10",
    )
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, gpu=GPU_CONFIG)
    .pip_install("httpx", "pynvml", "wandb")
)
```

Now that we've defined our image which we'll be using to embed our text, let's write up a small modal stub that we can call from within our main function to perform the embeddings.

``` python
# Define a class-based function with GPU configuration, Docker image, and concurrency settings
@stub.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    concurrency_limit=N_GPU,
    allow_concurrent_inputs=N_INPUTS,
)
class TextEmbeddingsInference:
    def __enter__(self):
        # Initialize the server and the HTTP client when the context is entered
        self.process = spawn_server()  # Start the server process
        self.client = AsyncClient(base_url="http://127.0.0.1:8000")  # Initialize an asynchronous HTTP client

    def __exit__(self, _exc_type, _exc_value, _traceback):
        # Terminate the server process when the context is exited
        self.process.terminate()

    @method()
    async def embed(self, texts: list[str]):
        n_chars = sum(map(len, texts))
        # Send a POST request to the /embed endpoint of the server, with the input texts in the request body
        _ = await self.client.post("/embed", json={"inputs": texts})
        # Return the total number of characters in the input texts
        return n_chars
```

Before we proceed to wire up our embedding function with our main function, let's highlight a few things that really make the experience with modal seamless when processing large batches of data

1. All it takes to provision a GPU is the variable `GPU_CONFIG`. What's even cooler is that you can specify multiple GPUs to be allocated to the same container, all it takes is to define a new variable as `GPU_CONFIG = gpu.A100(memory=80, count=2) `  and you suddenly have 160GB of interconnected RAM to run your huge 70b models and more!
2. You can run any arbitrary docker image and execute any code to create the perfect image for your container. Modal supports a whole range of functionality straight out of the box.
3. It's pretty straightforward to define teardown and setup functions by simple creating a `stub.cls` and writing a new `__enter__` function that'll be executed when your container gets called for the first time or a `__exit__` function that cleans everything up when your container is no longer in use.

## Wiring it all up

Now that we've got our local entrypoint to load our dataset and a container function defined which can generate embeddings given a list of texts, let's figure how to wire them together

```python
@stub.function(
    image=Image.debian_slim().pip_install("datasets"),
    volumes={cache_dir: volume},
    timeout=5000,
)
def embed_dataset(down_scale: float = 0.005, batch_size: int = 32):

  	# ... Other code before this
    batches = generate_batches(
        text_chunks, batch_size=batch_size
    )
    materialized_batches = list(batches)
  
    # Step 1 : Define an instance of our TextEmbeddingsInference class we defined earlier
    model = TextEmbeddingsInference()
    counter = 0
    for n_chars in model.embed.map(materialized_batches, order_outputs=False):
        counter += n_chars
    print(counter)
```

Earlier, we generated a list of chunks from the text in our dataset. We then grouped them into batches which contained `batch_size` chunks of ~400 characters each in the form of a generator. We can generate embeddings using the `.map` function that modal provides out of the box for us. 

This takes a list of inputs and generates additional instances of our `TextEmbeddingsInference`, each with their own respective A10G GPU attached. The input is streamed over to each container on demand, with Modal handling the provisioning, allocation of inputs and even the subsequent winding down of containers once they're no longer in use.

## Conclusion

In this article, we had a sneak peak of what we could accomplish with Modal's new Volume feature that allows us to read in large datasets as if they were on our local file system, loading in a 20GB dataset into memory in just under 5 seconds. We then used Modal's stub to define and provision instances of Text Embedding Inference images with attached GPUs with just 40 lines of code. Lastly, we ran a massive batch job on a subset of the dataset, with Modal handling the orchestration of the multiple instances with just a simple `.map` function provided out of the box.

In the next few articles, we'll look at how you can serve these embeddings at scale, deploy custom OSS models of your choice and even fine-tune these models based on real-time data ingestion. If you liked this article, make sure to follow Modal on twitter and jxnl for more content :) 