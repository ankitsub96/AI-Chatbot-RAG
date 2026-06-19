import json

# import openai
from app.clients.groq_client import client
from app.config.settings import MODEL
from app.utils.helpers import timer


def generate_response(
    messages,
    temperature=0,
    tools=None,
    tool_choice=None,
    max_tokens=None,
    response_format=None,
    # )-> openai.types.chat.ChatCompletion:
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

    if response_format:

        kwargs["response_format"] = response_format

    raw = client.chat.completions.with_raw_response.create(**kwargs)
    print(
        {
            "limit_requests": raw.headers.get("x-ratelimit-limit-requests"),
            "remaining_requests": raw.headers.get("x-ratelimit-remaining-requests"),
            "limit_tokens": raw.headers.get("x-ratelimit-limit-tokens"),
            "remaining_tokens": raw.headers.get("x-ratelimit-remaining-tokens"),
            "reset_requests": raw.headers.get("x-ratelimit-reset-requests"),
            "reset_tokens": raw.headers.get("x-ratelimit-reset-tokens"),
            "retry_after": raw.headers.get("retry-after"),
        }
    )
    return raw.parse()


@timer
def expand_query_sync(
    question: str,
    memory_context: str = "",
    n: int = 4,
) -> list[str]:
    prompt = f"""
    You are a query rewriting assistant for a retrieval system.

    Your goal is to generate search queries that retrieve evidence relevant to the ORIGINAL QUESTION.

    IMPORTANT:
    - Preserve the original meaning and intent.
    - Do NOT broaden the topic.
    - Do NOT introduce new questions.
    - Do NOT introduce related themes that are not explicitly present.
    - Keep important named entities, characters, places, events, and concepts.
    - Focus on alternative wording, terminology, and phrasing.
    - Queries should retrieve overlapping evidence from different lexical angles.
    - Avoid generic or high-level summaries.

    Conversation history:
    {memory_context or "None"}

    Original question:
    {question}

    Generate exactly {n} rewritten search queries.

    Good examples:

    Question:
    Why did Murtagh remain loyal to Galbatorix?

    Good rewrites:
    [
    "Reasons for Murtagh's loyalty to Galbatorix",
    "What motivated Murtagh to serve Galbatorix",
    "Factors influencing Murtagh's allegiance to Galbatorix",
    "Murtagh's relationship with Galbatorix and loyalty"
    ]

    Bad rewrites:
    [
    "Dragon rider politics",
    "Power struggles in Alagaësia",
    "History of Galbatorix",
    "Murtagh character analysis"
    ]

    Return ONLY a JSON array of strings.
    """

    try:
        response = generate_response(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        queries = json.loads(raw)

        if isinstance(queries, list):
            all_queries = [question]
            for q in queries:
                if isinstance(q, str) and q.strip() and q not in all_queries:
                    all_queries.append(q)
            print({"all_queries": all_queries})
            return all_queries[: n + 1]

    except Exception as e:
        print(f"[expand_query_sync] failed: {e}")

    return [question]
