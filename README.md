# Chatbridge

**Chatbridge** provides a locally hosted [OpenAI API compatible](https://platform.openai.com/docs/api-reference/introduction) endpoint that acts as an adaptor/proxy to [OCI Generative AI Inference APIs](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/).

Using Chatbridge enables you to configure AI clients that support adding OpenAI compatible models to use the Cohere and Meta Llama LLM models avaialble from OCI.

Chatbridge is a Python FastAPI based application that acts as a pass through between the OpenAI API endpoints to the equivilent OCI Generative AI API endpoints using the OCI Python SDK. Chatbridge supports the following APIs:

- `/v1/models` - list available chat models.
- `/v1/chat/completions` - chat completions.
- `/v1/embeddings` - uses an OCI supported embeddings model to generation to create the embedding vector representing the input text.

## Installation

Install from source into a local Python virtualenv. Reqiures the [`uv`](https://docs.astral.sh/uv/getting-started/installation/) pacakge manager.

```shell
git clone https://github.com/scross/Chatbridge
cd Chatbridge
uv venv
uv sync
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

- `API_KEY` - API key that must be passed by the client and the Authorization token (optional).

## Usage

><font color="#C93">⚠️ **CAUTION**</font>:
> Chatbridge uses your locally stored OCI credentails and is intended for localhost single user installation and access only. Chatbridge should not be used in a shared environment. Running the API on a non-local only IP exposes the API server to other machines on your network and potentially the internet. Anyone with access to the IP/URL will have direct authenticated access to you OCI Gen AI services. Ensure that you have appropriate security measures in place to limit access.

```shell
uvicorn Chatbridge.main:app --reload
```

By default the API will start on http://127.0.0.1:8000. To run on an alternative interface and port you can specify them as follows:

```shell
uvicorn Chatbridge.main:app --reload --host 127.0.0.1 --port 8080
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

Generate embeddings using a supported (embeddings model)(https://docs.oracle.com/en-us/iaas/Content/generative-ai/embed-models.htm)

#### `/docs`

Returns the FAST API documentation

## Debugging

The following environment variables can be used to enable debugging, set to `true` to enable. Default is `false`.

- `DEBUG` - Enable general debug logs
- `TRACE` - Enable additional trace level debug logs
- `DEBUG_OCI_SDK` - Enable OCI SDK debug logs
- `DEBUG_SSE_STARLETTE` - Enable SSE Starlette debug logs
