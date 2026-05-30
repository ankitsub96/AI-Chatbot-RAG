import json

from fastapi import APIRouter
from fastapi import Request

from app.models.extract_model import ExtractRequest

from app.services.rate_limit import limiter

from app.services.cache_service import redis_client

from app.services.ai_service import extract_ticket_data

from app.utils.helpers import create_cache_key

router = APIRouter()


@router.get("/")
def home():

    return {"message": "FastAPI working!"}


@router.post("/extract")
@limiter.limit("5/minute")
async def extract(request: Request, payload: ExtractRequest):

    texts = payload.texts

    combined_text = "||".join(texts)

    cache_key = create_cache_key(combined_text)

    cached = redis_client.get(cache_key)

    if cached:

        return {"cached": True, "data": json.loads(cached)}

    data = await extract_ticket_data(payload.texts)

    redis_client.setex(cache_key, 3600, json.dumps(data))

    return {"cached": False, "data": data}
