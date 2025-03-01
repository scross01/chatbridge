import oci
import uuid

from fastapi import FastAPI

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
        is_stream=False,  # TODO form_data.stream
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

    return OpenAIChatCompletionResponse(
        id=f"{response.data.model_id}-{str(uuid.uuid4())}",
        object="chat.completion",
        created=int(response.data.chat_response.time_created.timestamp()),
        model=response.data.model_id,
        choices=choices,
    )
