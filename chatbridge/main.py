import oci
import datetime
import dotenv
import json
import logging
import os
import uuid

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError

from sse_starlette import EventSourceResponse

from .schema import (
    OpenAIChatMessage,
    OpenAIChatChoice,
    OpenAIChatChunkChoice,
    OpenAIChatCompletionChunkResponse,
    OpenAIChatCompletionForm,
    OpenAIChatCompletionResponse,
    OpenAIChatDelta,
    OpenAIEmbeddingsForm,
    OpenAIEmbeddingsObject,
    OpenAIEmbeddingsResponse,
    OpenAPIFunctionCall,
    OpenAIModel,
)

dotenv.load_dotenv()

config_file = os.getenv("OCI_CONFIG_FILE", "~/.oci/config")
config_profile = os.getenv("OCI_CONFIG_PROFILE", "DEFAULT")
region = os.getenv("OCI_CONFIG_REGION", None)

# enables trace level logging
trace = os.getenv("TRACE", "false").lower() == "true"
# trace level logging will enable debug
debug = trace or os.getenv("DEBUG", "false").lower() == "true"
# enables OCI SDK debug logging
debug_oci = os.getenv("DEBUG_OCI_SDK", "false").lower() == "true"
# enables SSE Starlette library debug logging
debug_sse = os.getenv("DEBUG_SSE_STARLETTE", "false").lower() == "true"

# default logging settngs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# set component level logging based on the env settings
logger.setLevel(logging.DEBUG if debug else logging.INFO)
logging.getLogger("oci").setLevel(logging.DEBUG if debug_oci else logging.INFO)
logging.getLogger("sse_starlette").setLevel(
    logging.DEBUG if debug_sse else logging.INFO
)

api_key = os.getenv("API_KEY", None)
if not api_key or api_key == "":
    logger.warning(
        "API_KEY is not configured, access is open to unauthenticated clients."
    )

security = HTTPBearer()

app = FastAPI(
    debug=False,
    title="OpenAI Compatible API",
    description="""
    A FastAPI application that provides a local OpenAI compatible API interface to the OCI Generative AI services.
    """,
    version="0.1.0",
)

# load the OCI configuration profile
config = oci.config.from_file(file_location=config_file, profile_name=config_profile)
if region:
    # overide the region if set
    config["region"] = region

# create the OCI clients
generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config)
inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config
)


# Validate the API KEY if set
async def validate_api_key(
    credentials: None | HTTPAuthorizationCredentials = Depends(security),
):
    if not api_key or api_key == "":
        return None
    if credentials is None or credentials.scheme != "Bearer" or credentials.credentials != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


# Open AI compatible API endpoint to fetch list of supported models
# https://platform.openai.com/docs/api-reference/models
@app.get("/models")
@app.get("/v1/models")
async def get_models(api_key: str = Depends(validate_api_key)):

    logger.debug("/v1/models")

    resp = generative_ai_client.list_models(
        config["tenancy"],
        lifecycle_state="ACTIVE",
        capability=["CHAT"],  # NOTE: this filter has no effect - BUG?
        sort_by="displayName",  # NOTE: this option has no effect - BUG?
        sort_order="ASC",  # NOTE: this option has no effect - BUG?
    )

    if resp is None or resp.data is None:
        logger.error("Invalid or empty reponse")
        return {"error": {"code": None, "message": "empty response"}}

    # logger.debug(resp.data) if trace else None

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

    logger.debug(response) if trace else None

    return response


# Open AI compatible API endpoint for chat completions
# https://platform.openai.com/docs/api-reference/chat/create
@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    form_data: OpenAIChatCompletionForm, api_key: str = Depends(validate_api_key)
):

    logger.debug(f"/v1/chat/completions {form_data}")

    if form_data.model.startswith("meta."):
        return generic_chat_completions(form_data=form_data)
    elif form_data.model.startswith("cohere."):
        return cohere_chat_completions(form_data=form_data)
    elif form_data.model.startswith("xai."):
        return generic_chat_completions(form_data=form_data)
    else:
        # unsupported model type, try with the generic handler
        logger.warning(f"Unsupported model {form_data.model}")
        return generic_chat_completions(form_data=form_data)
        return


# Handle Cohere chat completions
def cohere_chat_completions(form_data: OpenAIChatCompletionForm):

    logger.info(
        f"Processing Cohere chat completion request using model {form_data.model}"
    )

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
        max_tokens=(
            form_data.max_completion_tokens
            if form_data.max_completion_tokens
            else form_data.max_tokens
        ),
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

    logger.debug(f"chat_details: {chat_details}") if trace else None

    try:
        resp = inference_client.chat(chat_details=chat_details)
    except oci.exceptions.ServiceError as se:
        logger.error(f"ServiceError {se}")
        return {"error": {"code": se.code, "message": se.message}}

    if resp is None or resp.data is None:
        logger.error("Invalid or empty reponse")
        return {"error": {"code": None, "message": "empty response"}}

    if form_data.stream:
        logger.info("Processing streaming response events")
        # re-stream the response
        return EventSourceResponse(cohere_restreamer(resp, form_data.model))  # type: ignore
    else:
        logger.info("Converting from Cohere response format")
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

        logger.debug(f"response: {response}") if trace else None

        return response


# Handle Generic chat completions (used for meta and xai models)
def generic_chat_completions(form_data: OpenAIChatCompletionForm):

    logger.info(
        f"Processing Generic chat completion request using model {form_data.model}"
    )

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
                elif item["type"] == "image_url":
                    image_url_data = item["image_url"]["url"]
                    content.append(
                        oci.generative_ai_inference.models.ImageContent(
                            type="IMAGE",
                            image_url=oci.generative_ai_inference.models.ImageUrl(
                                url=image_url_data,
                                detail=oci.generative_ai_inference.models.ImageUrl.DETAIL_AUTO,
                            ),
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
        max_tokens=(
            form_data.max_completion_tokens
            if form_data.max_completion_tokens
            else form_data.max_tokens
        ),
        top_k=form_data.top_k,
        top_p=form_data.top_p,
        frequency_penalty=form_data.frequency_penalty,
        presence_penalty=form_data.presence_penalty,
        stop=form_data.stop,
        logit_bias=form_data.logit_bias,
        log_probs=form_data.logprobs,
    )

    tools = []
    if form_data.tools:
        for tool in form_data.tools:
            tools.append(
                oci.generative_ai_inference.models.FunctionDefinition(
                    type="FUNCTION",
                    name=tool.function.name,
                    description=tool.function.description,
                    parameters=tool.function.parameters,
                )
            )
        chat_request.tools = tools

    chat_details = oci.generative_ai_inference.models.ChatDetails(
        compartment_id=config["tenancy"],
        serving_mode=serving_mode,
        chat_request=chat_request,
    )

    logger.debug(f"chat_details: {chat_details}") if trace else None

    try:
        resp = inference_client.chat(chat_details=chat_details)
    except oci.exceptions.ServiceError as se:
        logger.error(f"ServiceError {se}")
        return {"error": {"code": se.code, "message": se.message}}

    if resp is None or resp.data is None:
        logger.error("Invalid or empty reponse")
        return {"error": {"code": None, "message": "empty response"}}

    if form_data.stream:
        logger.info("Processing streaming response events")
        # re-stream the response
        return EventSourceResponse(generic_restreamer(resp, form_data.model))  # pyright: ignore[reportArgumentType]
    else:
        logger.info("Converting from Generic response format")
        logger.debug(f"response: {resp.data}") if trace else None

        # convert response format
        choices = []
        for choice in resp.data.chat_response.choices:
            content = choice.message.content[0].text if choice.message.content else None
            tool_calls = []
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    function = {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    tool_calls.append(
                        OpenAPIFunctionCall(
                            id=tool_call.id,
                            function=function,
                            type="function",
                        )
                    )

            choices.append(
                OpenAIChatChoice(
                    index=choice.index,
                    message=OpenAIChatMessage(
                        role=choice.message.role.lower(),
                        content=content,
                        tool_calls=tool_calls if len(tool_calls) > 0 else None,
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

        logger.debug(f"response: {response}") if trace else None

        return response


# Meta response re-streamer
# convert and re-stream the response event stream back to the client
# https://platform.openai.com/docs/api-reference/chat/streaming
async def generic_restreamer(response, model):
    logger.debug("Streaming response")
    try:
        content = ""
        id = f"chatcmpl-{str(uuid.uuid4())}"
        first_event = True
        for event in response.data.events():
            chunk = json.loads(event.data)
            # logger.debug(f"chunk: {chunk}") if trace else None
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

                # continue with same event based on event type
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
                    logger.debug(f'finish_reason: {chunk["finishReason"]}')
                    logger.debug(f"Streamed content: {content}") if trace else None
                    yield finish.model_dump_json()
                elif "toolCalls" in chunk["message"]:
                    # send tool call
                    tool_calls = []
                    index = 0
                    for tool_call in chunk["message"]["toolCalls"]:
                        function = dict()
                        if "name" in tool_call:
                            function["name"] = tool_call["name"]
                        if "arguments" in tool_call:
                            function["arguments"] = tool_call["arguments"]

                        tool_calls.append(
                            OpenAPIFunctionCall(
                                index=index,
                                id=tool_call["id"] if "name" in tool_call else None,
                                function=function,
                                type="function" if "name" in tool_call else None,
                            )
                        )
                        index += 1
                    # if the tool call is empty, set to None
                    if (
                        len(tool_calls) == 1
                        and tool_calls[0].id is None
                        and tool_calls[0].type is None
                        and tool_calls[0].function["arguments"] == ""
                    ):
                        tool_calls = None

                    message = OpenAIChatCompletionChunkResponse(
                        id=id,
                        object="chat.completion.chunk",
                        created=int(datetime.datetime.now().timestamp()),
                        model=model,
                        choices=[
                            OpenAIChatChunkChoice(
                                index=0, delta=OpenAIChatDelta(tool_calls=tool_calls)
                            )
                        ],
                    )
                    logger.debug(f"using tool calls: {tool_calls}") if trace else None
                    yield message.model_dump_json()
                elif "content" in chunk["message"]:
                    # send message content
                    text = chunk["message"]["content"][0]["text"]
                    content += text
                    message = OpenAIChatCompletionChunkResponse(
                        id=id,
                        object="chat.completion.chunk",
                        created=int(datetime.datetime.now().timestamp()),
                        model=model,
                        choices=[
                            OpenAIChatChunkChoice(
                                index=0,
                                delta=OpenAIChatDelta(
                                    content=text,
                                    finish_reason=None,
                                ),
                            )
                        ],
                    )
                    yield message.model_dump_json()

    except ValidationError as ve:
        logger.error(f"ValidationError {ve}, {event.data}")
    except oci.exceptions.ServiceError as se:
        logger.error(f"ServiceError {se}, {event.data}")


# Cohere response re-streamer
# convert and re-stream the response event stream back to the client
# https://platform.openai.com/docs/api-reference/chat/streaming
async def cohere_restreamer(response, model):
    logger.debug("Streaming response")
    try:
        content = ""
        id = f"chatcmpl-{str(uuid.uuid4())}"
        for event in response.data.events():
            chunk = json.loads(event.data)
            logger.debug(f"chunk: {chunk}") if trace else None
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
                logger.debug(f'finish_reason: {chunk["finishReason"]}')
                logger.debug(f"Streamed content: {content}") if trace else None
                yield finish.model_dump_json()
            elif "text" in chunk:
                # send content
                content += chunk["text"]
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
    except oci.exceptions.ServiceError as se:
        logger.error(f"ServiceError {se}, {event.data}")


# Open AI compatible API endpoint for embeddings
# https://platform.openai.com/docs/api-reference/embeddings
@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(
    form_data: OpenAIEmbeddingsForm, api_key: str = Depends(validate_api_key)
) -> OpenAIEmbeddingsResponse | None:

    logger.info(f"Processing request using model {form_data.model}")
    logger.debug(f"/v1/embeddings {form_data}")

    # When using embeddings for semantic search
    # - the search query should be embedded by setting input_type="search_query"
    # - the text passages that are being searched over should be embedded with input_type="search_document".
    input_type = (
        "SEARCH_DOCUMENT"
        if type(form_data.input[0]) is str
        and form_data.input[0].startswith("<document_metadata>")
        else "SEARCH_QUERY"
    )

    embed_text_details = oci.generative_ai_inference.models.EmbedTextDetails(
        inputs=[form_data.input] if type(form_data.input) is str else form_data.input,
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
            serving_type=oci.generative_ai_inference.models.ServingMode.SERVING_TYPE_ON_DEMAND,
            model_id=form_data.model,
        ),
        compartment_id=config["tenancy"],
        is_echo=False,
        truncate="END",  # “NONE”, “START”, “END”
        input_type=input_type,  # “SEARCH_DOCUMENT”, “SEARCH_QUERY”, “CLASSIFICATION”, “CLUSTERING”, “IMAGE”
    )

    logger.debug(f"embed_text_details = {embed_text_details}") if trace else None

    try:
        resp = inference_client.embed_text(embed_text_details=embed_text_details)
    except oci.exceptions.ServiceError as se:
        logger.error(f"ServiceError {se}")
        return

    if resp is None or resp.data is None:
        logger.error("Invalid or empty reponse")
        return

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

    logger.debug(f"response: {response}") if trace else None

    return response
