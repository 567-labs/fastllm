import modal
from main import app
from src.embeddings import get_model 

stub = modal.Stub("embedder")

def download_model():
  return get_model()

image = (
  modal.Image.debian_slim()
  .pip_install(
    "fastapi",
    "sentence-transformers",
    "pydantic",
  )
  .run_function(download_model)
)

@stub.function(image=image)
@modal.asgi_app()
def fastapi_app():
  return app