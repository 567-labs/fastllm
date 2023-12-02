from modal import Image, Stub, Volume


vol = Volume.persisted("wikipedia")

stub = Stub("wikipedia")
stub.volume = vol


@stub.function(
    image=Image.debian_slim().pip_install("tqdm", "datasets"),
    volumes={"/data": vol},
    timeout=300,
)
def embed_dataset():
    # >>> modal run download.py::embed_dataset
    from datasets import load_dataset, load_from_disk

    WIKI, SET = (
        "wikipedia",
        "20220301.frr",  # "20210301.en" but its too big for testings
    )
    PATH = f"/data/{WIKI}-{SET}"

    # check if dataset is already downloaded
    try:
        dataset = load_from_disk(PATH)
        print("Dataset found, loading...")
        return dataset
    except FileNotFoundError:
        print("Dataset not found, downloading...")
        pass
    dataset = load_dataset(WIKI, SET)
    dataset.save_to_disk(PATH)
    stub.volume.commit()

    print(f"{len(dataset)=}")
