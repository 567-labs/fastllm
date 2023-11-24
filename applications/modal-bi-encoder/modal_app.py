import modal
from src.main import get_model
from app import app

stub = modal.Stub("bi-encoder")


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
