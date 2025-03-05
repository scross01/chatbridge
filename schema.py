# from https://github.com/open-webui/pipelines/blob/main/schemas.py

from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class OpenAIChatMessage(BaseModel):
    role: str
    content: str | List

    model_config = ConfigDict(extra="allow")


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

    model_config = ConfigDict(extra="allow")


class OpenAIModel(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: str | None


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChoice]


class OpenAIChatDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    finish_reason: Optional[str] = None


class OpenAIChatChunkChoice(BaseModel):
    index: int
    delta: OpenAIChatDelta


class OpenAIChatCompletionChunkResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatChunkChoice]
