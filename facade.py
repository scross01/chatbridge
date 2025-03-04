import oci
import datetime
import json
import uuid

from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse


from schema import (
    OpenAIChatMessage,
    OpenAIChatChoice,
    OpenAIChatCompletionForm,
    OpenAIChatCompletionResponse,
    OpenAIModel,
)

app = FastAPI()

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

    response = generative_ai_client.list_models(
        config["tenancy"],
        lifecycle_state="ACTIVE",
        capability=["CHAT"],  # NOTE: this filter has no effect - BUG?
        sort_by="displayName",  # NOTE: this option has no effect - BUG?
        sort_order="ASC",  # NOTE: this option has no effect - BUG?
    )

    # filter the response
    # limit to only the active chat models
    # TODO getting multiple models with the same name
    chat_models = []
    for model in response.data.items:
        if model.lifecycle_state == "ACTIVE" and "CHAT" in model.capabilities:
            item = OpenAIModel(
                id=model.display_name,
                object="model",
                created=int(model.time_created.timestamp()),
                owned_by=model.vendor,
            )
            chat_models.append(item)

    return {
        "object": "list",
        "data": chat_models,
    }


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
    print(form_data)

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
        is_stream=False,  # TODO form_data.stream
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

    print(chat_request)

    response = inference_client.chat(chat_details=chat_details)

    print(response.data)

    # convert response format.
    # only use the last item from the chat history
    choices = []
    item = response.data.chat_response.chat_history[-1]
    choices.append(
        OpenAIChatChoice(
            message=OpenAIChatMessage(
                role=item.role.lower() if item.role != "CHATBOT" else "assistant",
                content=item.message,
            ),
            finish_reason=response.data.chat_response.finish_reason,
            index=0,
        )
    )

    print(choices)

    return OpenAIChatCompletionResponse(
        id=f"{response.data.model_id}-{str(uuid.uuid4())}",
        object="chat.completion",
        created=int(datetime.datetime.now().timestamp()),
        model=response.data.model_id,
        choices=choices,
    )


def meta_chat_completions(form_data: OpenAIChatCompletionForm):
    print(form_data)

    # convert message format
    messages = []
    for message in form_data.messages:
        messages.append(
            oci.generative_ai_inference.models.UserMessage(
                role=message.role.upper(),
                content=[
                    oci.generative_ai_inference.models.TextContent(
                        type="TEXT",
                        text=message.content,
                    )
                ],
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

    print(chat_request)

    response = inference_client.chat(chat_details=chat_details)

    if form_data.stream:
        # re-stream the response
        return EventSourceResponse(restreamer(response, form_data.model))
    else:
        print(response.data)
        # convert response format
        choices = []
        for choice in response.data.chat_response.choices:
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
        print(choices)
        return OpenAIChatCompletionResponse(
            id=f"{response.data.model_id}-{str(uuid.uuid4())}",
            object="chat.completion",
            created=int(response.data.chat_response.time_created.timestamp()),
            model=response.data.model_id,
            choices=choices,
        )


# response re-streamer
# https://platform.openai.com/docs/api-reference/chat/streaming
async def restreamer(response, model):
    try:
        id = f"chatcmpl-{str(uuid.uuid4())}"
        first_event = True
        for event in response.data.events():
            print(event.data)
            chunk = json.loads(event.data)

            if "message" in chunk:
                if first_event:
                    # send role event
                    message = json.dumps({  # TODO convert to schema object
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": int(datetime.datetime.now().timestamp()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": chunk["message"]["role"].lower(),
                                "finish_reason": None,
                            }
                        }]
                    })
                    first_event = False
                    print(message)
                    yield message

                # send content
                message = json.dumps({  # TODO convert to schema object
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": chunk["message"]["content"][0]["text"],
                            "finish_reason": None,
                        }
                    }]
                })
                print(message)
                yield message
            elif "finishReason" in chunk:
                finish = json.dumps({  # TODO convert to schema object
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": chunk["finishReason"],
                    }]
                })
                print(finish)
                yield finish
    except:  # TODO
        print("Exception",  event.data)
        pass
