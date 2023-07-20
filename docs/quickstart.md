
## Setting the application

After installing `fastllm`, you have two options: start up the FastAPI server or run the docker container.

The two options are described as follows:

### Starting up the FastAPI server

This is the default option. To do this, you can run the following command:

```shell
uvicorn run:app --reload
```

The server will be running at `http://localhost:8000`. You can then interact with the server using the `curl` command or a web browser.

### Running the docker container

This is an alternative option that allows you to run fastllm in a docker container. To do this, you can run the following command:

```bash
docker run -d -p 8000:8000 jxnl/embed-gpt4all:latest
```

!!! note

    The step above assumes you have pulled the Docker image from the Docker Hub.


## Running the application

Send a POST request to the `/v1/embedding` endpoint:

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
