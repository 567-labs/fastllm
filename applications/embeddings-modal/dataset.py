import requests
import datasets
import time
import subprocess
import pandas as pd
from typing import List

# Replace with your equivalent Modal Labs endpoint
MODAL_ENDPOINT = "https://jumshim--embedder-fastapi-app-dev.modal.run/embed"
DATASET = {
  "name": "wikipedia",
  "subset": "20220301.simple",
}
EXCEL_FILE = "embeddings.xlsx"


# Wrapper Benchmarker to evaluate runtime of each function
def benchmark(func):
  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Time taken for {func.__name__}: {end_time - start_time} seconds")
    return result
  return wrapper

@benchmark
def start_modal_server():
  subprocess.run(["modal", "serve", "modal_main.py"])

@benchmark
def get_embeddings(texts: List[str]) -> List[List[float]]:
  response = requests.post(
    MODAL_ENDPOINT,
    json={"input": texts}
  )
  response.raise_for_status()
  return [item["embedding"] for item in response.json()["data"]]

def process_dataset(dataset):
  embeddings = []
  for i, item in enumerate(dataset):
    if (i >= 1000):
      break
    embedding = get_embeddings([item['text']])
    embeddings.append(embedding)
  return embeddings

def save_to_excel(data):
  df = pd.dataframe(data)
  df.to_excel(EXCEL_FILE, index=False)
  print(f"Data saved to: {EXCEL_FILE}")

def main():
  dataset = datasets.load_dataset(DATASET["name"], DATASET["subset"])['train']
  start_time = time.time()
  embeddings = process_dataset(dataset)
  end_time = time.time()
  save_to_excel(embeddings)
  print("------------PROCESS END--------------")
  print(f"Time taken for processing and embedding: {end_time - start_time} seconds")

if __name__ == "__main__":
  main()

