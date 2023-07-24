import modal
from main import app, model

stub = modal.Stub("cross-encoder")


def download_model():
    return model()


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
