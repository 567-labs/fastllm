# Modal Bi-Encoder Embedding Service

This is an app to run a [bi-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html#bi-encoder-vs-cross-encoder) sentence transformer model to create text embeddings.

This application is capable or running locally as well as running on [Modal](https://modal.com/). It also enables fine tuning the Bi-Encoder model both locally and on Modal.

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com//fastllm.git
    ```

2. Navigate to the project directory:

    ```shell
    cd applications/modal-bi-encoder
    ```

3. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

## Usage

### Local

1. Start the FastAPI server:

    ```shell
    uvicorn app:app --reload
    ```

    The server will be running at `http://localhost:8000`.

2. Run any of the following sample POST request commands

    **Encode:**

    ```shell
    curl -X POST -H "Content-Type: application/json" -d '{"texts": ["hello world", "hi there"]}' localhost:8000/encode
    ```

    expected result: {"embeddings":[[0.034345343708992004,0.03316108137369156,0.021912500262260437, ... ]]}

    **Cosine Similarity:**

    ```shell
    curl -X POST -H "Content-Type: application/json" -d '{"text1": "hello world", "text2": "hi earth"}' localhost:8000/cosine_similarity
    ```

    expected result: {"similarity":0.6782615184783936}

    **TODO: finetune**

### Modal

1. Create a [Modal](https://modal.com/) account if you haven't already, and install the modal CLI

    ```shell
    pip install modal
    ```

2. Deploy the modal app

    ```shell
    modal serve modal_app.py
    ```

3. Run any of the following sample commands. Your <MODAL_APP_URL> can be found in your [Modal Running Apps](https://modal.com/apps)

    **Encode:**

    ```shell
    curl -X POST -H "Content-Type: application/json" -d '{"texts": ["hello world", "hi there"]}' <MODAL_APP_URL>/encode
    ```

    expected result: {"embeddings":[[0.034345343708992004,0.03316108137369156,0.021912500262260437, ... ]]}

    **Cosine Similarity:**

    ```shell
    curl -X POST -H "Content-Type: application/json" -d '{"text1": "hello world", "text2": "hi earth"}' <MODAL_APP_URL>/cosine_similarity 
    ```

    expected result: {"similarity":0.6782615184783936}

    **TODO: finetune**
