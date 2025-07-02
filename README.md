# Chatbridge

**Chatbridge** provides a locally hosted [OpenAI API compatible](https://platform.openai.com/docs/api-reference/introduction) endpoint that acts as an adaptor/proxy to [OCI Generative AI Inference APIs](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/).

Using Chatbridge enables you to configure AI clients that support adding OpenAI compatible models to use the Generatei AI LLM models avaialble from Oracle Cloud Infrastructure.  

>You must have a paid OCI account with access to the specific regions that host the Generative AI services. Generative AI models are not availale for OCI Always-Free Tier accounts.

Chatbridge is a Python FastAPI based application that acts as a pass through between the OpenAI API endpoints to the equivilent OCI Generative AI API endpoints using the OCI Python SDK. Chatbridge supports the following APIs:

- `/v1/models` - list available chat models.
- `/v1/chat/completions` - chat completions.
- `/v1/embeddings` - uses an OCI supported embeddings model to generation to create the embedding vector representing the input text.

## Installation

Install from source into a local Python virtualenv. Reqiures the [`uv`](https://docs.astral.sh/uv/getting-started/installation/) pacakge manager.

```shell
git clone https://github.com/scross/chatbridge
cd chatbridge
uv sync
source .venv/bin/activate
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

## Chatbridge configuration

The following additional configuration seetings can also be set in the local shell or the `.env` file.

- `API_KEY` - API key that must be passed by the client as the Authorization token (optional). The API key can be any text string. One way to generate a new unique key is to run `openssl rand -base64 32`.

## Usage

><font color="#C93">⚠️ **CAUTION**</font>:
> Chatbridge uses your locally stored OCI credentails and is intended for localhost single user installation and access only. Chatbridge should not be used in a shared environment. Running the API on a non-local only IP exposes the API server to other machines on your network and potentially the internet. Anyone with access to the IP/URL will have direct authenticated access to you OCI Gen AI services. Ensure that you have appropriate security measures in place to limit access including setting a unique `API_KEY`.

```shell
uvicorn chatbridge.main:app --reload
```

By default the API will start on `http://127.0.0.1:8000`. To run on an alternative interface and port you can specify them as follows:

```shell
uvicorn chatbridge.main:app --host 127.0.0.1 --port 8080
```

### Running using Docker

A [Dockerfile](./Dockerfile) and sample [docker-compose.yml](./docker/docker-compose.yml) are included for running Chatbridge using docker.

```shell
docker build . --tag chatbridge
```

To run using `docker run`. Ensure that `API_KEY` is set in your `.env` file.

```shell
docker run --rm -p 8000:8000 -v ~/.oci:/home/chatbridge/.oci --env-file .env chatbridge 
```

To run using docker compose. Modify the `docker/docker-compose.yml` to set a new `API_KEY`

```shell
cd docker
docker compose up
```

## Supported APIs and capabilities

#### `/v1/models`

List of supported [chat models](https://docs.oracle.com/en-us/iaas/Content/generative-ai/chat-models.htm). The results are filtered only include models that have the "CHAT" capability.

Note the OCI API response appears to include a few LLM CHAT models that are not actually available, or not available in the selected region.

#### `/v1/chat/completions`

Generate chat completions for Meta and Cohere models. Automatically uses the appropriate OCI Inference API for Cohere [CohereChatRequest](https://docs.oracle.com/iaas/api/#/en/generative-ai-inference/latest/datatypes/CohereChatRequest) or Meta Llama [GenericChatRequest](https://docs.oracle.com/iaas/api/#/en/generative-ai-inference/latest/datatypes/GenericChatRequest) based on the model selection.

Supported [OpenAI API chat completion options](https://platform.openai.com/docs/api-reference/chat/create)

| Capability | Meta | Cohere |
| - | - | - |
| audio | x | x |
| frequency_penalty | ✅ | ✅ |
| logit_bias | ✅ | x |
| logprobs | ✅ | x |
| max_tokens (deprecated) | ✅ | ✅ |
| max_completion_tokens | ✅ | ✅ |
| metadata | x | x |
| modalities | x | x |
| n | x | x |
| parallel_tool_calls | x | x |
| prediction | x | x |
| presence_penalty | ✅ | ✅ |
| reasoning_effort | x | x |
| response_format | x | x |
| seed | ✅ | ✅ |
| service_tier | x | x |
| stop | ✅ | ✅ |
| store | x | x |
| stream | ✅ | ✅ |
| stream_options | x | x |
| temperature | ✅ | ✅ |
| tool_choice | x | x |
| tools | ✅ | x |
| top_logprobs | x | x |
| top_k | ✅ | ✅ |
| top_p | ✅ | ✅ |
| user | x | x |
| web_search_options | x | x |

#### `/v1/embeddings`

Generate embeddings using a supported [embeddings model](https://docs.oracle.com/en-us/iaas/Content/generative-ai/embed-models.htm)

#### `/docs`

Returns the FAST API documentation

## Supported Models

Refer to the list of OCI [Pretrained Foundational Models in Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm#pretrained-models) for the full list of availabe models and supported regions.

Chatbridge currently only works with **On-Demand** models, support for dedicated AI clusters is not implemented. Note that not all models are availale on-demand in each region.

The following models have been tested in us-chicago-1.

| Chat Models | Status | Comments |
| - | - | - |
| cohere.command-a-03-2025 | ✅ | |
| cohere.command-r-08-2024 | ✅ | |
| cohere.command-r-plus-08-2024 | ✅ | |
| cohere.command-r-16k | x | retired |
| cohere.command-r-plus | x | retired |
| meta.llama-4-maverick-17b-128e-instruct-fp8 | ✅ | |
| meta.llama-4-scout-17b-16e-instruct | ✅ | |
| meta.llama-3.3-70b-instruct | ✅ | |
| meta.llama-3.2-90b-vision-instruct | ✅ | |
| meta.llama-3.2-11b-vision-instruct | x | dedicated AI clusters only |
| meta.llama-3.1-405b-instruct | ✅ | |
| meta.llama-3.1-70b-instruct | ✅ | deprecated |
| meta.llama-3-70b-instruct | x | retired |
| xai.grok-3 | ? | not tested |
| xai.grok-3-mini | ? | not tested |
| xai.grok-3-fast | ? | not tested |
| xai.grok-3-mini-fast | ? | not tested |

| Embedding Models | Status | Comments |
| - | - | - |
| cohere.embed-english-image-v3.0 | x | dedicated AI clusters only |
| cohere.embed-english-light-image-v3.0 | x | dedicated AI clusters only |
| cohere.embed-multilingual-light-image-v3.0 | x | dedicated AI clusters only |
| cohere.embed-english-v3.0 | ✅ | |
| cohere.embed-multilingual-v3.0 | ✅ | |
| cohere.embed-english-light-v3.0 | ✅ | |
| cohere.embed-multilingual-light-v3.0 | ✅ | |

| Reranking Models | Status | Comments |
| - | - | - |
| cohere.rerank.3-5 | x | dedicated AI clusters only |

## Supported Clients

Chatbridge should work with any AI Client that supports an Open AI compatible model with option to set the custom API URL. Charbridge has been tested with the following clients.

| Client | Status | Comments |
| - | - | - |
| [Cherry Studio](https://www.cherry-ai.com/) | ✅ | In the Mode Provider settings add a new provider called "OCI Generative AI", set the Provider Type to "OpenAI". Set the API Key and API Host |
| [fabric](https://github.com/danielmiessler/fabric) | 🟡 | The **LLM Studio** configuration option can be used with the Chatbridge endpoint, but only works if the API Key is not enabled. |
| [Open WebUI](https://docs.openwebui.com/) | ✅ | In the Connections settings an a new OpenAI API Connection. Set the connection URL and Key. Optionally set the Prefix ID to "OCI". |
| [Roo Code](https://roocode.com/) | ✅ | Create a new Provider using the OpenAI Compatible option. Set the Base URL, API Key, and Model. Disable the "Include max output tokens" option unless a specific Max Output Tokens value is being set. |

## Debugging

The following environment variables can be used to enable debugging, set to `true` to enable. Default is `false`.

- `DEBUG` - Enable general debug logs
- `TRACE` - Enable additional trace level debug logs
- `DEBUG_OCI_SDK` - Enable OCI SDK debug logs
- `DEBUG_SSE_STARLETTE` - Enable SSE Starlette debug logs

## Development

Run the server with the `--reload` option to automitally pickup code changes.

```shell
uvicorn chatbridge.main:app --reload
```
