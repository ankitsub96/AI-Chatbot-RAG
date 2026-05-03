import json
import asyncio

from app.services.llm_service import generate_response
from app.schemas.tools import tools


async def extract_single_ticket(text: str):

    response = await generate_response(
        messages=[
            {
                "role": "system",
                "content": """
                You extract structured information from:
                - support tickets
                - emails
                - bug reports

                Return:
                - intent
                - entities
                - priority_score
                - suggested_action

                Priority score rules:

                1-3:
                - minor inconvenience
                - general questions
                - feature requests
                - cosmetic/UI issues

                4-6:
                - standard bugs
                - partial functionality issues
                - affecting one user or small group

                7-8:
                - important production problems
                - billing/payment issues
                - login/authentication failures
                - repeated customer impact

                9-10:
                - critical outages
                - security incidents
                - data loss
                - production systems down
                - widespread customer impact

                Always return:
                - concise intent
                - meaningful entities
                - realistic priority score
                - actionable suggested_action
                """,
            },
            {"role": "user", "content": text},
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    tool_call = response.choices[0].message.tool_calls[0]

    return {
        "success": True,
        "input": text,
        "data": json.loads(tool_call.function.arguments),
    }


async def extract_ticket_data(texts: list[str]):

    tasks = [extract_single_ticket(text) for text in texts]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    formatted_results = []

    for result in results:

        if isinstance(result, Exception):

            formatted_results.append({"success": False, "error": str(result)})

        else:

            formatted_results.append(result)

    return formatted_results
