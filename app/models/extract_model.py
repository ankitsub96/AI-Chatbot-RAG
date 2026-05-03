from pydantic import BaseModel
from pydantic import Field


class ExtractRequest(BaseModel):

    texts: list[str] = Field(
        ...,
        example=[

            # Priority 2
            "How can I export my account data to CSV?",

            # Priority 4
            "Customer profile picture upload failing on mobile app",

            # Priority 6
            "Several users report delayed email notifications after signup",

            # Priority 8
            "Customer unable to login after password reset",

            # Priority 9
            "Payment dashboard crashing in production for multiple users",

            # Priority 10
            "Production database outage causing complete checkout failure across all regions"
        ]
    )
