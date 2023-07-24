import modal
from main import app, model

stub = modal.Stub("jsonformer")


def download_model():
    return model()


image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "transformers",
        "jsonformer",
        "torch",
        "bitsandbytes>=0.39.0",
        "accelerate",
        "einops",
        "scipy",
        "numpy",
    )
    .run_function(download_model)
)


@stub.function(image=image, gpu="any")
@modal.asgi_app()
def fastapi_app():
    return app
