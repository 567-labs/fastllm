# Modal / Base Ten Embedding Service

This is a project comparing Modal and BaseTen services. Each will host a FastAPI app that provides an embedding service. It accepts text input and returns the corresponding embeddings.

## Installation

1. Clone the repository:

```shell
git clone https://github.com//fastllm.git
```

2. Navigate to the project directory:

```shell
cd applications/embeddings-modal
```

3. Install the required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:

```shell
uvicorn main:app --reload
```

The server will be running at `http://localhost:8000`.

2. Send a POST request to the `/embed` endpoint:

```shell
curl -X POST -H "Content-Type: application/json" -d '{"input": "The food was delicious and the waiter...", "model": ""BAAI/bge-base-en-v1.5" }' http://localhost:8000/embed
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
    },
    ],
    "model": "BAAI/bge-base-en-v1.5",
    "usage": {
        "prompt_tokens": 0,
        "total_tokens": 0
    }
}
```

The `embedding` field contains the embedding values for the input text.

## Prebuilt Docker

### Step 1: Pull the Docker Image

To start using a Docker image from Docker Hub, follow these steps:

1. Open a terminal or command prompt.

2. Run the following command to pull the desired Docker image from Docker Hub:

```bash
docker pull jxnl/embed-gpt4all
```

This will download the `xnl/embed-gpt4all` image from Docker Hub with the "latest" tag.

### Step 2: Run the Docker Container

Once the Docker image is pulled, you can run a container based on that image:

```bash
docker run -d -p 8000:8000 jxnl/embed-gpt4all:latest
```

This command will run the `jxnl/embed-gpt4all` container in detached mode (`-d`) and forward port 8000 on the host to port 8000 inside the container.

### Accessing the Application

After running the container, you can access the application running inside the container through your web browser or any HTTP client by visiting `http://localhost:8000`.

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

### Building your own image

Once you have cloned the repository, navigate to the root directory of the application where the Dockerfile is located.

To build the Docker image, use the `docker build` command. The syntax is as follows:

```bash
docker build -t <image_name>:<tag> .
```

- `<image_name>`: Choose a meaningful name for your Docker image, e.g., `my-app`.
- `<tag>`: Specify a version or tag for your image, e.g., `v1.0`.

Replace `<image_name>` and `<tag>` with your desired values. The `.` at the end of the command refers to the current directory, which is where the Dockerfile resides.

Example:

```bash
docker build -t my-app:v1.0 .
```

The Docker build process will read the Dockerfile instructions and create an image containing your application and its dependencies.

### Running the Container

Once the Docker image is built, you can run a container based on that image:

```bash
docker run -d -p <host_port>:8000 <image_name>:<tag>
```

- `<host_port>`: The port number on your host machine where you want to access the application.
- `8000`: The port number exposed by the application inside the Docker container.

Example:

```bash
docker run -d -p 8000:8000 my-app:v1.0
```

## Modal Usage

1. Navigate to the project directory:

```shell
cd applications/embeddings-modal
```

2. Start the Modal server:

```shell
modal serve modal_main.py
```
