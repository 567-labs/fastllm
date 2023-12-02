from modal import Image, Volume, Stub


cache_dir = "/data"

dataset_name = "gsm8k"
dataset_config = "main"

volume = Volume.persisted("embedding-wikipedia")
image = Image.debian_slim().pip_install("datasets")


stub = Stub(image=image)


@stub.function(volumes={cache_dir: volume})
def list_all_files():
    import os

    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            size = os.path.getsize(fp)
            print(f"File: {fp}, Size: {size}")


# Set a really high timeout
@stub.function(volumes={cache_dir: volume}, timeout=3000)
def download_dataset(cache=False):
    # Redownload the dataset
    from datasets import load_dataset
    import time

    start = time.time()
    dataset = load_dataset("wikipedia", "20220301.en", num_proc=10)
    end = time.time()
    print(f"Download complete - downloaded files in {end-start}s")
    if cache:
        dataset.save_to_disk(f"{cache_dir}/wikipedia")

        volume.commit()


@stub.function(volumes={cache_dir: volume})
def check_dataset_exists():
    volume.reload()
    from datasets import load_from_disk

    print("Loading dataset from disk...")
    dataset = load_from_disk(f"{cache_dir}/wikipedia")
    print(dataset)  # Print out a quick summary of the dataset
    print("Dataset loaded.")


@stub.local_entrypoint()
def main():
    # list_all_files.remote()
    download_dataset.remote()
    # check_dataset_exists.remote()
