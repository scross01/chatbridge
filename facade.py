import oci
import datetime
import json
import logging
import sys
import uuid

from fastapi import FastAPI
from pydantic import ValidationError

# from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse

from schema import (
    OpenAIChatMessage,
    OpenAIChatChoice,
    OpenAIChatCompletionForm,
    OpenAIChatCompletionResponse,
    OpenAIEmbeddingsObject,
    OpenAIModel,
    OpenAIChatDelta,
    OpenAIChatChunkChoice,
    OpenAIChatCompletionChunkResponse,
    OpenAIEmbeddingsForm,
    OpenAIEmbeddingsResponse,
)

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# XXX
profile_name = "ocicpm"
region = "uk-london-1"

# TODO get from config file or environment
config = oci.config.from_file(profile_name=profile_name)
config["region"] = region
generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config)
inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config
)


# Open AI compatible API endpoint to fetch list of supported models
# https://platform.openai.com/docs/api-reference/models
@app.get("/models")
@app.get("/v1/models")
async def get_models():

    resp = generative_ai_client.list_models(
        config["tenancy"],
        lifecycle_state="ACTIVE",
        capability=["CHAT"],  # NOTE: this filter has no effect - BUG?
        sort_by="displayName",  # NOTE: this option has no effect - BUG?
        sort_order="ASC",  # NOTE: this option has no effect - BUG?
    )

    logger.debug(resp.data)

    # filter the response
    # limit to only the active chat models
    chat_models = []
    for model in resp.data.items:
        # Note: filter for exactly ["CHAT"], not just contains "CHAT",
        # otherwise model names are repeated.
        if model.lifecycle_state == "ACTIVE" and model.capabilities == ["CHAT"]:
            item = OpenAIModel(
                id=model.display_name,
                object="model",
                created=int(model.time_created.timestamp()),
                owned_by=model.vendor,
            )
            chat_models.append(item)

    response = {
        "object": "list",
        "data": chat_models,
    }

    logger.debug(chat_models)

    return response


# Open AI compatible API endpoint for chat completions
# https://platform.openai.com/docs/api-reference/chat/create
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(form_data: OpenAIChatCompletionForm):

    if form_data.model.startswith("meta."):
        return meta_chat_completions(form_data=form_data)
    elif form_data.model.startswith("cohere."):
        return cohere_chat_completions(form_data=form_data)
    else:
        # unsupported model type
        return


def cohere_chat_completions(form_data: OpenAIChatCompletionForm):

    logger.debug(form_data)

    serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        serving_type="ON_DEMAND",
        model_id=form_data.model,
    )

    chat_history = []
    # collect the message history, excluding the last one which is the user's input
    for message in form_data.messages[:-1]:
        if message.role == "system":
            chat_history.append(
                oci.generative_ai_inference.models.CohereSystemMessage(
                    message=message.content
                )
            )
        elif message.role == "user":
            chat_history.append(
                oci.generative_ai_inference.models.CohereUserMessage(
                    message=message.content
                )
            )
        elif message.role == "assistant":
            chat_history.append(
                oci.generative_ai_inference.models.CohereChatBotMessage(
                    message=message.content
                )
            )
        else:
            raise ValueError(f"Unsupported role: {message.role}")

    prompt = form_data.messages[-1].content

    chat_request = oci.generative_ai_inference.models.CohereChatRequest(
        message=prompt,
        chat_history=chat_history,
        response_format=oci.generative_ai_inference.models.CohereResponseTextFormat(),
        is_stream=form_data.stream,
        seed=form_data.seed,
        temperature=form_data.temperature,
        max_tokens=form_data.max_tokens,
        top_k=form_data.top_k,
        top_p=form_data.top_p,
        frequency_penalty=form_data.frequency_penalty,
        presence_penalty=form_data.presence_penalty,
        stop_sequences=form_data.stop,
        # documents=[],
        # is_search_queries_only=,
        # preamble_override=,
        # max_input_tokens=,
        # prompt_truncation=, # "OFF", "AUTO_PRESERVE_ORDER"
        # is_echo=,
        # tools=,
        # tool_results=,
        # is_force_single_step=,
        # is_raw_prompting=,
        # citation_quality=, # "ACCURATE","FAST"
    )

    chat_details = oci.generative_ai_inference.models.ChatDetails(
        compartment_id=config["tenancy"],
        serving_mode=serving_mode,
        chat_request=chat_request,
    )

    resp = inference_client.chat(chat_details=chat_details)

    if form_data.stream:
        # re-stream the response
        return EventSourceResponse(cohere_restreamer(resp, form_data.model))
    else:
        # convert response format.
        # only use the last item from the chat history
        choices = []
        item = resp.data.chat_response.chat_history[-1]
        choices.append(
            OpenAIChatChoice(
                message=OpenAIChatMessage(
                    role=item.role.lower() if item.role != "CHATBOT" else "assistant",
                    content=item.message,
                ),
                finish_reason=resp.data.chat_response.finish_reason,
                index=0,
            )
        )

        response = OpenAIChatCompletionResponse(
            id=f"{resp.data.model_id}-{str(uuid.uuid4())}",
            object="chat.completion",
            created=int(datetime.datetime.now().timestamp()),
            model=resp.data.model_id,
            choices=choices,
        )

        logger.info(response)

        return response


def meta_chat_completions(form_data: OpenAIChatCompletionForm):
    # convert message format
    messages = []

    for message in form_data.messages:
        content = []
        if type(message.content) is str:
            content.append(
                oci.generative_ai_inference.models.TextContent(
                    type="TEXT",
                    text=message.content,
                )
            )
        elif type(message.content) is list:
            for item in message.content:
                if item["type"] == "text":
                    content.append(
                        oci.generative_ai_inference.models.TextContent(
                            type="TEXT",
                            text=item["text"],
                        )
                    )
                else:
                    logger.error(f'unsupported message content type {item["type"]}')
        else:
            logger.error(
                f"unexpected message content colletion {type(message.content)}"
            )

        messages.append(
            oci.generative_ai_inference.models.UserMessage(
                role=message.role.upper(),
                content=content,
            )
        )

    serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
        serving_type="ON_DEMAND",
        model_id=form_data.model,
    )

    chat_request = oci.generative_ai_inference.models.GenericChatRequest(
        api_format="GENERIC",
        messages=messages,
        is_stream=form_data.stream,
        seed=form_data.seed,
        temperature=form_data.temperature,
        max_tokens=form_data.max_tokens,
        top_k=form_data.top_k,
        top_p=form_data.top_p,
        frequency_penalty=form_data.frequency_penalty,
        presence_penalty=form_data.presence_penalty,
        stop=form_data.stop,
    )

    chat_details = oci.generative_ai_inference.models.ChatDetails(
        compartment_id=config["tenancy"],
        serving_mode=serving_mode,
        chat_request=chat_request,
    )

    resp = inference_client.chat(chat_details=chat_details)

    if form_data.stream:
        # re-stream the response
        return EventSourceResponse(meta_restreamer(resp, form_data.model))
    else:
        # convert response format
        choices = []
        for choice in resp.data.chat_response.choices:
            choices.append(
                OpenAIChatChoice(
                    index=choice.index,
                    message=OpenAIChatMessage(
                        role=choice.message.role.lower(),
                        content=choice.message.content[0].text,
                    ),
                    finish_reason=choice.finish_reason,
                )
            )

        response = OpenAIChatCompletionResponse(
            id=f"{resp.data.model_id}-{str(uuid.uuid4())}",
            object="chat.completion",
            created=int(resp.data.chat_response.time_created.timestamp()),
            model=resp.data.model_id,
            choices=choices,
        )

        logger.debug(response)
        return response


# response re-streamer
# https://platform.openai.com/docs/api-reference/chat/streaming
async def meta_restreamer(response, model):
    try:
        id = f"chatcmpl-{str(uuid.uuid4())}"
        first_event = True
        for event in response.data.events():
            chunk = json.loads(event.data)

            if "message" in chunk:
                if first_event:
                    # send just the role first as a separate chunk
                    message = OpenAIChatCompletionChunkResponse(
                        id=id,
                        object="chat.completion.chunk",
                        created=int(datetime.datetime.now().timestamp()),
                        model=model,
                        choices=[
                            OpenAIChatChunkChoice(
                                index=0,
                                delta=OpenAIChatDelta(
                                    role=chunk["message"]["role"].lower(),
                                    finish_reason=None,
                                ),
                            )
                        ],
                    )
                    first_event = False
                    yield message.model_dump_json()

                # send message content
                message = OpenAIChatCompletionChunkResponse(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(datetime.datetime.now().timestamp()),
                    model=model,
                    choices=[
                        OpenAIChatChunkChoice(
                            index=0,
                            delta=OpenAIChatDelta(
                                content=chunk["message"]["content"][0]["text"],
                                finish_reason=None,
                            ),
                        )
                    ],
                )
                yield message.model_dump_json()
            elif "finishReason" in chunk:
                finish = OpenAIChatCompletionChunkResponse(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(datetime.datetime.now().timestamp()),
                    model=model,
                    choices=[
                        OpenAIChatChunkChoice(
                            index=0,
                            delta=OpenAIChatDelta(),
                            finish_reason=chunk["finishReason"],
                        ),
                    ],
                )
                yield finish.model_dump_json()
    except ValidationError as ve:
        logger.error(f"ValidationError {ve}, {event.data}")
    except:  # TODO
        e = sys.exc_info()[0]
        logger.error(f"Exception {e}, {event.data}")
        pass


async def cohere_restreamer(response, model):
    try:
        id = f"chatcmpl-{str(uuid.uuid4())}"
        for event in response.data.events():
            chunk = json.loads(event.data)
            if "finishReason" in chunk:
                finish = OpenAIChatCompletionChunkResponse(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(datetime.datetime.now().timestamp()),
                    model=model,
                    choices=[
                        OpenAIChatChunkChoice(
                            index=0,
                            delta=OpenAIChatDelta(),
                            finish_reason=chunk["finishReason"],
                        ),
                    ],
                )
                yield finish.model_dump_json()
            elif "text" in chunk:
                # send content
                message = OpenAIChatCompletionChunkResponse(
                    id=id,
                    object="chat.completion.chunk",
                    created=int(datetime.datetime.now().timestamp()),
                    model=model,
                    choices=[
                        OpenAIChatChunkChoice(
                            index=0,
                            delta=OpenAIChatDelta(
                                content=chunk["text"],
                                finish_reason=None,
                            ),
                        )
                    ],
                )
                yield message.model_dump_json()
    except ValidationError as ve:
        logger.error(f"ValidationError {ve}, {event.data}")
    except:  # TODO
        e = sys.exc_info()[0]
        logger.error(f"Exception {e}, {event.data}")
        pass


# Open AI compatible API endpoint for embeddings
# https://platform.openai.com/docs/api-reference/embeddings
@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(form_data: OpenAIEmbeddingsForm) -> OpenAIEmbeddingsResponse:

    logger.debug(form_data)

    # When using embeddings for semantic search
    # - the search query should be embedded by setting input_type="search_query"
    # - the text passages that are being searched over should be embedded with input_type="search_document".
    input_type = (
        "SEARCH_DOCUMENT"
        if form_data.input[0].startswith("<document_metadata>")
        else "SEARCH_QUERY"
    )

    embed_text_details = oci.generative_ai_inference.models.EmbedTextDetails(
        inputs=form_data.input if type(form_data.input) == list else [form_data.input],
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
            serving_type=oci.generative_ai_inference.models.ServingMode.SERVING_TYPE_ON_DEMAND,
            model_id=form_data.model,
        ),
        compartment_id=config["tenancy"],
        is_echo=False,
        truncate="END",  # “NONE”, “START”, “END”
        input_type=input_type,  # “SEARCH_DOCUMENT”, “SEARCH_QUERY”, “CLASSIFICATION”, “CLUSTERING”, “IMAGE”
    )

    resp = inference_client.embed_text(embed_text_details=embed_text_details)

    logger.debug(resp.data)

    # Extract embeddings from response
    embeddings = resp.data.embeddings

    response = OpenAIEmbeddingsResponse(
        object="list",
        data=[
            OpenAIEmbeddingsObject(
                object="embedding",
                index=0,
                embedding=embeddings[0],
            )
        ],
        model=resp.data.model_id,
        usage={},
    )

    logger.debug(response)

    return response
