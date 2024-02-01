from typing import Literal, List
from tqdm import tqdm
import os
from cohere import Client
from openai import AsyncOpenAI
import asyncio
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from helpers.cache import cache_res

model_types = Literal["OpenAI", "Cohere", "HuggingFace"]
MODEL_TYPES_VALUES = {"HuggingFace", "OpenAI", "Cohere"}


class EmbeddingModel:
    def __init__(self, model_name: str, model_type: model_types, max_limit: 20):
        """
        This is an abstract model class which provides an easy to use interface for benchmarking different embedding providers.

        Currently it supports any arbitrary hugging face, openai and cohere embedding models. Note that the `.embed` function accepts a generator
        """
        from asyncio import Semaphore

        self.model_name = model_name
        self.model_type: model_types = model_type
        self.semaphore = Semaphore(max_limit)

        assert (
            self.model_type in MODEL_TYPES_VALUES
        ), f"Invalid model type: {self.model_type}. Expected one of {model_types}"

    @cache_res
    @retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=10, max=240))
    async def _embed_cohere(self, texts: List[str]):
        async with self.semaphore:
            co = Client(os.environ["COHERE_API_KEY"])
            response = co.embed(
                texts=texts,
                model=self.model_name,
                input_type="clustering",
            )
            self.tqdm_monitoring_bar.update(len(texts))
            return response.embeddings

    @cache_res
    @retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=10, max=240))
    async def _embed_openai(self, texts: List[str]):
        async with self.semaphore:
            client = AsyncOpenAI()
            try:
                response = await client.embeddings.create(
                    input=texts, model=self.model_name
                )
                self.tqdm_monitoring_bar.update(len(texts))
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"Error occurred while creating embeddings: {e}")
                print(texts)
                raise e

    async def embed(self, texts: List[str], batch_size):
        self.tqdm_monitoring_bar = tqdm(total=batch_size)

        self.tqdm_monitoring_bar.set_description_str(
            f"Embedding Progress for {self.model_name}"
        )
        if self.model_type == "Cohere":
            coros = [self._embed_cohere(sentence_group) for sentence_group in texts]
            res = await asyncio.gather(*coros)
            return [item for sublist in res for item in sublist]

        if self.model_type == "OpenAI":
            coros = [self._embed_openai(sentence_group) for sentence_group in texts]
            res = await asyncio.gather(*coros)
            return [item for sublist in res for item in sublist]

        if self.model_type == "HuggingFace":
            embeddings = []
            model = SentenceTransformer(self.model_name)
            ttl = 0
            for item in texts:
                embeddings.extend(model.encode(item))
                self.tqdm_monitoring_bar.update(len(item))
                ttl += len(item)
            return embeddings
