from typing import List, Optional
from pydantic import BaseModel, ConfigDict


# simplified representation of OpenAI Chat API response
# https://platform.openai.com/docs/api-reference/chat/create
class OpenAIChatMessage(BaseModel):
    role: str
    content: str | List

    model_config = ConfigDict(extra="allow")


# OpenAI function definition object
# https://platform.openai.com/docs/api-reference/chat/create
class OpenAIFunctionDefinition(BaseModel):
    name: str
    description: Optional[str]
    parameters: Optional[dict]
    strict: Optional[bool] = False


# OpenAI tool definition object
# https://platform.openai.com/docs/api-reference/chat/create
class OpenAIToolDefintion(BaseModel):
    function: OpenAIFunctionDefinition
    type: str


# representation of an OpenAI Chat request
# https://platform.openai.com/docs/api-reference/chat/create
class OpenAIChatCompletionForm(BaseModel):
    stream: bool = True
    model: str
    messages: List[OpenAIChatMessage]

    # optional attributes
    seed: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[OpenAIToolDefintion]] = None
    
    model_config = ConfigDict(extra="allow")


# representation of the OpenAI Model response data item
# https://platform.openai.com/docs/api-reference/models
class OpenAIModel(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


# OpenAI chat completion response data for choices items
# https://platform.openai.com/docs/api-reference/chat/object#chat/object-choices
class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: str | None


# OpenAI chat completion response object
# https://platform.openai.com/docs/api-reference/chat/object
class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChoice]


# OpenAI chat completion streaming delta for tool calls
# https://platform.openai.com/docs/guides/function-calling#streaming
class OpenAPIFunctionCall(BaseModel):
    # [{"index": 0, "id": "call_DdmO9pD3xa9XTPNJ32zg2hcA", "function": {"arguments": "", "name": "get_weather"}, "type": "function"}]
    # [{"index": 0, "id": null, "function": {"arguments": "{\"", "name": null}, "type": null}]
    index: int
    id: str | None
    function: dict
    type: str | None


# OpenAI chat completion streaming detla object within response choices
# https://platform.openai.com/docs/api-reference/chat/streaming#chat/streaming-choices
class OpenAIChatDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[OpenAPIFunctionCall]] = None


# OpenAI chat completion streaming response choices
# https://platform.openai.com/docs/api-reference/chat/streaming#chat/streaming-choices
class OpenAIChatChunkChoice(BaseModel):
    index: int
    delta: OpenAIChatDelta
    finish_reason: Optional[str] = None


# OpenAI chat completion streaming response object
# https://platform.openai.com/docs/api-reference/chat/streaming
class OpenAIChatCompletionChunkResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChunkChoice]


# OpenAI embeddings form data
# https://platform.openai.com/docs/api-reference/embeddings/create
class OpenAIEmbeddingsForm(BaseModel):
    input: str | List[str] | List[float]
    model: str
    encoding_format: Optional[str] = "float"

    model_config = ConfigDict(extra="allow")


# OpenAI embeddings object
# https://platform.openai.com/docs/api-reference/embeddings/object
class OpenAIEmbeddingsObject(BaseModel):
    object: str
    embedding: List[float]
    index: int


# OpenAI embeddings response data
# https://platform.openai.com/docs/api-reference/embeddings/create
class OpenAIEmbeddingsResponse(BaseModel):
    object: str
    data: List[OpenAIEmbeddingsObject]
    model: str
    usage: dict
