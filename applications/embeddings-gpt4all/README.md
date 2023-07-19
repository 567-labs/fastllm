# FastAPI Embedding Service

This is a FastAPI app that provides an embedding service. It accepts text input and returns the corresponding embeddings.

## Installation

1. Clone the repository:

```shell
git clone https://github.com//fastllm.git
```

2. Navigate to the project directory:

```shell
cd applications/embeddings-gpt4all
```

3. Install the required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:

```shell
uvicorn run:app --reload
```

The server will be running at `http://localhost:8000`.

2. Send a POST request to the `/v1/embedding` endpoint:

```shell
curl -X POST -H "Content-Type: application/json" -d '{"input": "The food was delicious and the waiter..."}' http://localhost:8000/v1/embedding
```

The server will respond with the embeddings in the following format:

```json
{
    "object": "list",
    "data": [
    {
        "object": "embedding",
        "embedding": [0.0023064255, -0.009327292, ... ],
        "index": 0
    }
    ]
}
```

The `embedding` field contains the embedding values for the input text.

## Docker - FILLME

To run this in docker you must be in this directory

```sh
docker build -t fastapi-embedding .
```

This command builds the Docker image with the tag `fastapi-embedding``. Make sure to include the . at the end of the command, indicating that the build context is the current directory.

Once the image is built, you can run a container using the image:

```sh
docker run -d -p 8000:8000 fastapi-embedding
```

This command starts a container in detached mode (-d) and maps the container's port 8000 to the host's port 8000. Adjust the port mapping if necessary.

The FastAPI app will now be accessible at http://localhost:8000 from your local machine.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT)