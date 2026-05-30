tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_ticket_info",
            "description": "Extract structured support ticket data",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string"
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "priority_score": {
                        "type": "integer"
                    },
                    "suggested_action": {
                        "type": "string"
                    }
                },
                "required": [
                    "intent",
                    "entities",
                    "priority_score",
                    "suggested_action"
                ]
            }
        }
    }
]