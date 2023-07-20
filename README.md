# fastllm

A collection of LLM services you can self host via docker or modal labs to support your applications development
## Objectives

The goal of this repo is to provide a series of docker containers, or modal labs deployments of common patterns when using LLMs and provide endpoints that allows you to intergrate easily with existing codebases that use the popular openai api.

## Roadmap

* Support GPT4all's embedding api and match it to `openai.com/v1/embedding`
* Support JSONFormer api to match it to chatcompletion with function_calls
* Support Cross Encoders based on sentence transformers for any huggingface model
* Provide great documentation and runbooks using MkDocs
## Prerequisites

Before you begin, ensure that you have the following installed on your system:

1. Git: Visit the [Git website](https://git-scm.com/) to download and install Git on your operating system.
2. Docker: Visit the [Docker website](https://www.docker.com/) to download and install Docker on your operating system.

### Clone the Repository

To clone the repository to your local machine, follow these steps:

1. Open a terminal or command prompt.

2. Change the current directory to where you want to store the repository.

3. Run the following command to clone the repository:

```bash
git clone https://github.com/jxnl/fastllm.git
```
This will create a local copy of the repository on your machine.

### Building the Docker Image

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
### Accessing the Application

After running the container, you can access the application running inside the container through your web browser or any HTTP client by visiting `http://localhost:8000`.


## Pull Prebuild dockerImage

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

### Stopping and Removing Containers

To stop a running container, use the `docker stop` command followed by the container ID or name:

```bash
docker stop <container_id_or_name>
```

To remove a stopped container, use the `docker rm` command followed by the container ID or name:

```bash
docker rm <container_id_or_name>
```
### Stopping and Removing Containers

To stop a running container, use the `docker stop` command followed by the container ID or name:

```bash
docker stop <container_id_or_name>
```

To remove a stopped container, use the `docker rm` command followed by the container ID or name:

```bash
docker rm <container_id_or_name>
```

### Cleaning Up

To remove the Docker image, use the `docker rmi` command followed by the image ID or name:

```bash
docker rmi <image_id_or_name>
```

Remember that removing an image is irreversible and cannot be undone.
