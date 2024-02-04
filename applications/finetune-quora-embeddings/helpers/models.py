from typing import Literal, List
from tqdm import tqdm
import os
from cohere import Client
from openai import AsyncOpenAI
from diskcache import Cache
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

model_types = Literal["OpenAI", "Cohere", "HuggingFace"]
MODEL_TYPES_VALUES = {"HuggingFace", "OpenAI", "Cohere"}


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        model_type: model_types,
        max_limit: int = 20,
        expected_iterations: int = float("inf"),
    ):
        """
        This is an abstract model class which provides an easy to use interface for benchmarking different embedding providers.

        Currently it supports any arbitrary hugging face, openai and cohere embedding models. Note that the `.embed` function accepts a generator
        """
        from asyncio import Semaphore

        self.model_name = model_name
        self.model_type: model_types = model_type
        self.semaphore = Semaphore(max_limit)
        self.cache = Cache(os.getcwd())
        self.expected_iterations = expected_iterations

        assert (
            self.model_type in MODEL_TYPES_VALUES
        ), f"Invalid model type: {self.model_type}. Expected one of {model_types}"

    @classmethod
    def from_hf(cls, model_name: str, expected_iterations: int):
        return cls(
            model_name,
            model_type="HuggingFace",
            max_limit=float("inf"),
            expected_iterations=expected_iterations,
        )

    @classmethod
    def from_openai(cls, model_name: str, max_limit=20):
        return cls(model_name, model_type="OpenAI", max_limit=max_limit)

    @classmethod
    def from_cohere(cls, model_name: str):
        return cls(model_name, model_type="Cohere")

    def _embed_cohere(self, texts: List[str]):
        import json
        from cohere.client import EmbedJob

        co = Client(os.environ["COHERE_API_KEY"])
        jsonl_file = "texts.jsonl"

        with open(jsonl_file, "w") as file:
            for text in texts:
                json.dump({"text": text}, file)
                file.write("\n")

        input_dataset = co.create_dataset(
            name="embed-job",
            data=open(jsonl_file, "rb"),
            dataset_type="embed-input",
        )
        input_dataset.await_validation()
        embed_job: EmbedJob = co.create_embed_job(
            dataset_id=input_dataset.id,
            input_type="clustering",
            model="embed-english-v3.0",
            truncate="END",
        )
        embed_job.wait()
        output_dataset = co.get_dataset(embed_job.output.id)
        results = []
        for record in output_dataset:
            results.append(record["embedding"])
        return results

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=8, max=240))
    async def _embed_openai(self, texts: List[str]):
        key = texts[0]
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value

        async with self.semaphore:
            client = AsyncOpenAI()
            try:
                response = await client.embeddings.create(
                    input=texts, model=self.model_name
                )
                embeddings = [item.embedding for item in response.data]
                self.cache.set(key, embeddings)
                return embeddings
            except Exception as e:
                print(f"Error occurred while creating embeddings: {e}")
                raise e

    async def embed(self, texts: List[str]):
        from tqdm.asyncio import tqdm_asyncio

        if self.model_type == "Cohere":
            for sentence_group in texts:
                return self._embed_cohere(sentence_group)

        if self.model_type == "OpenAI":
            coros = [self._embed_openai(sentence_group) for sentence_group in texts]
            results = await tqdm_asyncio.gather(*coros)
            return [item for sublist in results for item in sublist]

        if self.model_type == "HuggingFace":
            embeddings = []
            model = SentenceTransformer(self.model_name)
            for item in tqdm(texts, total=self.expected_iterations):
                embeddings.extend(model.encode(item))
            return embeddings
