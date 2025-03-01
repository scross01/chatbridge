# from https://github.com/open-webui/pipelines/blob/main/schemas.py

from typing import List
from pydantic import BaseModel, ConfigDict


class OpenAIChatMessage(BaseModel):
    role: str
    content: str | List

    model_config = ConfigDict(extra="allow")


class OpenAIChatCompletionForm(BaseModel):
    stream: bool = True
    model: str
    messages: List[OpenAIChatMessage]

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
