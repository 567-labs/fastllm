# FastAPI Function Calls Service

This is a FastAPI app that provides an function call service. It accepts the same openai input for chat completions
and returns the corresponding embeddings.

## Installation

1. Clone the repository:

```shell
git clone https://github.com//fastllm.git
```

2. Navigate to the project directory:

```shell
cd applications/jsonformer
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

2. Send a POST request to the `/v1/chat/completions` endpoint:

```shell
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "databricks/dolly-v2-7b",
  "messages": [
    {"role": "user", "content": "What is the weather like in Boston?"}
  ],
  "functions": [
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  ]
}'
```

The server should response with the following:

```json
{
  "id": "chatcmpl-123",
  ...
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_current_weather",
        "arguments": "{ \"location\": \"Boston, MA\"}"
      }
    },
    "finish_reason": "function_call"
  }]
}
```

## Docker - FILLME

To run this in docker you must be in this directory

```sh
docker build -t fastapi-jsonformer .
```

This command builds the Docker image with the tag `fastapi-embedding``. Make sure to include the . at the end of the command, indicating that the build context is the current directory.

Once the image is built, you can run a container using the image:

```sh
docker run -d -p 8000:8000 fastapi-jsonformer 
```

This command starts a container in detached mode (-d) and maps the container's port 8000 to the host's port 8000. Adjust the port mapping if necessary.

The FastAPI app will now be accessible at http://localhost:8000 from your local machine.

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT)