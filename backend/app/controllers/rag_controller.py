import os

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    BackgroundTasks,
)
from fastapi.responses import StreamingResponse

from app.services.rag_service import (
    build_vector_database,
    list_ready_documents,
    ask_document,
)
from app.services.langchain_rag_service import ask_document_langchain

from app.models.rag_model import AskDocumentRequest

from app.models.session_model import SearchMemoryRequest

from app.config.settings import UPLOAD_DIR

from app.utils.file_utils import (
    generate_file_hash,
    get_index_path,
    get_metadata_path,
)

from app.services.memory_service import (
    get_all_sessions,
    get_session_history,
    semantic_search_session,
    delete_session_memory,
)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    # -------------------------
    # READ FILE
    # -------------------------

    file_bytes = await file.read()

    # -------------------------
    # GENERATE DETERMINISTIC NAME
    # -------------------------

    file_hash = generate_file_hash(file_bytes)

    file_size = len(file_bytes)

    base_name = os.path.splitext(file.filename)[0]
    stored_filename = f"{base_name}_{file_hash}_{file_size}.pdf"
    # -------------------------
    # CHECK IF ALREADY INDEXED
    # -------------------------

    index_path = get_index_path(stored_filename)

    metadata_path = get_metadata_path(stored_filename)

    if os.path.exists(index_path) and os.path.exists(metadata_path):

        return {
            "message": "File already indexed",
            "status": "ready",
            "original_filename": file.filename,
            "stored_filename": stored_filename,
        }

    # -------------------------
    # SAVE FILE LOCALLY
    # -------------------------

    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    with open(file_path, "wb") as f:

        f.write(file_bytes)

    # -------------------------
    # BACKGROUND INDEXING
    # -------------------------

    background_tasks.add_task(build_vector_database, stored_filename)

    # -------------------------
    # RESPONSE
    # -------------------------

    return {
        "message": "File uploaded",
        "status": "processing",
        "original_filename": file.filename,
        "stored_filename": stored_filename,
    }


@router.get("/documents")
async def documents():

    return {"documents": list_ready_documents()}


@router.post("/ask")
async def ask_pdf(payload: AskDocumentRequest, background_tasks: BackgroundTasks):

    answer = await ask_document(
        payload.filenames, payload.question, payload.session_id, background_tasks
    )

    return {"answer": answer}


@router.post("/ask/langchain")
async def ask_pdf_langchain(
    payload: AskDocumentRequest,
    background_tasks: BackgroundTasks,
    stream: bool = False,
):

    result = await ask_document_langchain(
        payload.filenames,
        payload.question,
        payload.session_id,
        background_tasks,
        stream=stream,
    )

    # =========================
    # STREAM MODE
    # =========================
    if stream:

        # ensure it's actually a generator
        if not hasattr(result, "__aiter__"):
            raise ValueError("Expected async generator for streaming mode")

        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # =========================
    # NORMAL MODE
    # =========================
    return {"answer": result}


@router.get("/sessions")
async def sessions():

    return {"sessions": get_all_sessions()}


@router.get("/sessions/{session_id}")
async def session_history(session_id: str, page: int = 1, page_size: int = 20):

    history = get_session_history(session_id, page, page_size)

    return {"session_id": session_id, "history": history}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):

    delete_session_memory(session_id)

    return {"message": "Session deleted"}


@router.post("/sessions/{session_id}/search")
async def search_session_memory(session_id: str, payload: SearchMemoryRequest):

    results = semantic_search_session(session_id, payload.query)

    return {"results": results}
