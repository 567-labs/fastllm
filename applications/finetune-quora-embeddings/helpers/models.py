import enum
import os
from typing import List
from tqdm import tqdm
from cohere import Client
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class Provider(enum.Enum):
    HUGGINGFACE = "HuggingFace"
    OPENAI = "OpenAI"
    COHERE = "Cohere"


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        provider: Provider,
        max_limit: int = 20,
        expected_iterations: int = float("inf"),
    ):
        """
        This is an abstract model class which provides an easy to use interface for benchmarking different embedding providers.

        Currently it supports any arbitrary hugging face, openai and cohere embedding models. Note that the `.embed` function accepts a generator
        """
        from asyncio import Semaphore

        self.model_name = model_name
        self.provider = provider
        self.semaphore = Semaphore(max_limit)
        self.expected_iterations = expected_iterations

    @classmethod
    def from_hf(cls, model_name: str):
        return cls(
            model_name,
            provider=Provider.HUGGINGFACE,
            max_limit=float("inf"),
        )

    @classmethod
    def from_openai(cls, model_name: str, max_limit=20):
        return cls(model_name, provider=Provider.OPENAI, max_limit=max_limit)

    @classmethod
    def from_cohere(cls, model_name: str):
        return cls(model_name, provider=Provider.COHERE)

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
        async with self.semaphore:
            client = AsyncOpenAI()
            try:
                response = await client.embeddings.create(
                    input=texts, model=self.model_name
                )
                embeddings = [item.embedding for item in response.data]
                return embeddings
            except Exception as e:
                print(f"Error occurred while creating embeddings: {e}")
                raise e

    async def embed(self, texts: List[str]):
        from tqdm.asyncio import tqdm_asyncio

        if self.provider == Provider.COHERE:
            sentence_group = texts[0]  # Cohere only supports 1 batch of sentences
            print(f"Using Cohere with {len(sentence_group)} sentences")
            return self._embed_cohere(sentence_group)

        if self.provider == Provider.OPENAI:
            texts = [text for text in texts if len(text) > 0]
            print(
                f"Using OpenAI {self.model_name} with {len(texts)} batchs of sentences"
            )
            coros = [self._embed_openai(sentence_group) for sentence_group in texts]
            results = await tqdm_asyncio.gather(*coros)
            return [item for sublist in results for item in sublist]

        if self.provider == Provider.HUGGINGFACE:
            texts = [text for text in texts if len(text) > 0]
            print(
                f"Using HuggingFace {self.model_name} with {len(texts)} batchs of sentences"
            )
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            model.to("cuda")
            embeddings = model.encode(texts, show_progress_bar=True)
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Embeddings: {embeddings}")
            return embeddings
