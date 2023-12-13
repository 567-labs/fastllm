# Embedding Wikipedia under 30 minutes

For many companies, increased user growth often comes with various complications around scaling their core product offering. Modal offers an easy solution here - allowing companies to scale compute on demand to meet production needs. In fact all it takes is to add in a simple python decorator and you'll be able to even do something as crazy as embed all of English Wikipedia in just under 30 minutes. 


In case you're unfamiliar with Modal, they're a Infrastructure as Code (IAC) company that aims to simplify the complex process of deploying your code. By only paying for what you need, and abstracting away all the complexity of deploying and serving, Modal provides a simplified process to help you focus on what's important - the features that you want to build.

> To follow along with some of these examples, you'll need to create a Modal account at their https://modal.com. You'll get $30 out of the box and all of the features immediately to try out. Once you've done so, make sure to install the `modal` python package using a virtual environment of your choice and you can run all of the code we've provided below.

Modal makes working with open source embedding models a breeze - all you need to do to load in models or datasets is to store it in one of their volumes or create a docker image with the data in mind. For companies, this means being able to avoid issues such as arbitrary rate limits while simultaneously opening up new possibilities for more customised and fine-tuned models that can be used in production for better performance. 

In short, go from spending days tuning your pytorch code with esoteric vectorization technioques and move towards running massive compute jobs by adding a simple `.map` to your code while letting Modal handle all the network orchestration and compute allocation.

# Concepts 

With that in mind, let's start with something simpler - how do we even bring our existing code and datasets into Modal and run a simple job? In order to understand that, we'll need to look at two concepts - a Stub and a Volume.

## Stubs

A Stub is the most basic concept in Modal. We can declare it using just 2 lines of code.

```python
import modal

stub = modal.stub()
```

Once we've declared the stub, you can then do really interesting things such as provision GPUs to run workloads, instantiate an endpoint to serve large language models at scale and even spin up hundreds of containers to process large datasets in parallel. 

**It's as simple as declaring a new property on your stub.**

Stubs can even be run with any arbitrary docker image, pinned dependencies and custom dockerfiles with the custom Rust build system Modal provides. All it takes is to define the image using Modal's `Image` object, adding it to the stub declaration and you're good to go.


## Volumes

Now that we've figured out how to spin up Modal containers, how can we get our models or datasets into these containers efficiently and quickly? 

> Remember, a 70B model is going to be __almost 130GB__ in size, if you're looking to download that using a simple curl command, you're going to be waiting for a very long time. 

We could use something like S3 but that means that for large datasets and weights, we'd be eating significant network eggress charges while simultaneously dealing with slow transfer speeds and rate limits.

That's where Volumes come in - they allow us to have our target data right beside our GPU and load it into our containers concurrently. The feature itself is built to allow for all of your containers to read in data quickly and efficiently at the same time. This is ideal for jobs where we might need to run a large batch job involving a distributed dataset or a model whose weights we'd like to load in quickly. 

It's suprisingly simple to set-up, all you need to create a new volume is the following 5 lines of code

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

In some of our initial test runs, we were able to load in the entirety of English Wikipedia, all 22 GB of it, into a container in just under 5 seconds and start executing jobs. Compare that to the 10 minutes it took to download the dataset from scratch and you can imagine the huge benefits this would bring for any endpoints or batch jobs down the line. 



# Embedding Wikipedia

Now that we've got a good understanding of some key concepts that Modal provides, let's look at volumes in action by running our first embedding job. We've loaded a small dataset into a directory called `wikipedia` inside a persistent volume we've created called `embedding-wikipedia`

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

With these two new lines of code, we can cut down the time taken by almost 90%, from 4 hours to 30 minutes without any need for any complex optimisations or specialised code.

# Conclusion

In today's Article, we went through a few foundational concepts that are key to taking advantage of Modal's full capabilities - namely stubs, volumes and how they can be used in tandem to run massive parallelizable jobs at scale. 

This unlocks new business use cases for companies that can now iterate on production models more quickly and efficiently. By shortening the feedback loop with Modal's serverless soluitions, companies are then freed up to focus on what matters  - their product and new features.

We've uploaded our code [here](link). You can also check out some datasets [here](https://huggingface.co/567-labs) containing embeddings that we computed using some popular open source embedding models.
