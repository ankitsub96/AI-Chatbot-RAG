import redis

from app.config.settings import REDIS_HOST
from app.config.settings import REDIS_PORT


redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)