from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlmodel import SQLModel

from app.controllers.extract_controller import router
from app.controllers.rag_controller import router as rag_router
from app.services.rate_limit import limiter
from app.services.rate_limit import rate_limit_handler
from app.services.database import engine

from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# models must be imported before create_all so SQLModel registers them
from app.models.conversation_memory import ConversationMemory
from app.models.conversation_summary import ConversationSummary
from app.models.document_chunk import DocumentChunk
from app.models.session import Session
from app.models.document import Document
from app.models.session_document import SessionDocument

app = FastAPI()

app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)

app.add_exception_handler(RateLimitExceeded, rate_limit_handler)


@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


app.include_router(router)

app.include_router(rag_router)


# ... all your existing routers ...

# serve frontend static files
app.mount("/assets", StaticFiles(directory="../frontend/dist/assets"), name="assets")


@app.get("/app")
def serve_frontend():
    return FileResponse("../frontend/dist/index.html")
