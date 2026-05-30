from app.clients.groq_client import client
from app.config.settings import MODEL


def generate_response(
    messages,
    temperature=0,
    tools=None,
    tool_choice=None,
    max_tokens=None,
):

    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:

        kwargs["tools"] = tools

    if tool_choice:

        kwargs["tool_choice"] = tool_choice

    if max_tokens:

        kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**kwargs)

    return response
