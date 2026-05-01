from fastapi import FastAPI

from app.controllers.extract_controller import router
from app.controllers.rag_controller import (
    router as rag_router
)
from app.services.rate_limit import limiter
from app.services.rate_limit import rate_limit_handler

from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

app = FastAPI()

app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)

app.add_exception_handler(
    RateLimitExceeded,
    rate_limit_handler
)

app.include_router(router)

app.include_router(rag_router)