# OCI GenAI OpenAI Compatible API Adapter

A FastAPI application that provides a local OpenAI compatible API interface to the OCI Generative AI service.

This application allows you to interact with the OCI Generative AI service using a locally hosted API interface, which can be useful quickly connect AI tools and clients that support OpenAI's API interface.

## Installation

```python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### OCI API Configuration

Before running the application, you need to configure your OCI access credentials on your local machine. If you have not done this before you can run

```shell
oci setup bootstrap
```

 The configuration file is typically located at `~/.oci/config`. You can also specify and alternavice config file location, profile name, and region using environment variables or a local `.env` file.

- `OCI_CONFIG_FILE`: Path to the OCI configuration file (default: `~/.oci/config`)
- `OCI_CONFIG_PROFILE`: Name of the OCI configuration profile (default: `DEFAULT`)
- `OCI_CONFIG_REGION`: Region where the OCI service is located (optional)

## Usage

```shell
uvicorn main:app --reload
```

By default the API will start on http://127.0.0.1:8000. To run on an alternative interface and port you can specify them as follows:

```shell
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

> CAUTION: running the API on 0.0.0.0 exposes the API server to other machines on your local network and potentially the internet. Anyone with access to the URL will have direct access to you OCI Gen AI service. Ensure that you have appropriate security measures in place to limit access.


## Supported APIs

- `/v1/models` - List of supported [https://docs.oracle.com/en-us/iaas/Content/generative-ai/chat-models.htm](chat models)
- `/v1/chat/completions` - Generate chat completions for Meta and Cohere models
- `/v1/embeddings` - Generate embeddings using a supported [https://docs.oracle.com/en-us/iaas/Content/generative-ai/embed-models.htm](embeddings model)

- `/docs` - API documentation

## Roadmap

- TODO add support for images
- TODO add support for documents and RAG
- TODO add support for tool calling
- TODO add command line options
- TODO add API token bearer authentication
