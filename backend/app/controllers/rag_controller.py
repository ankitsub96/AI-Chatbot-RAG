import os

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.services.rag_service import (
    build_vector_database,
    list_ready_documents,
    ask_document,
    create_session,
    get_session_documents,
    unlink_session_document,
    delete_session_full,
    cleanup_orphan_documents,
)
from app.services.langchain_rag_service import ask_document_langchain
from app.models.rag_model import AskDocumentRequest, AgentAskRequest
from app.models.session_model import SearchMemoryRequest
from app.models.document import Document
from app.models.session_document import SessionDocument
from app.config.settings import UPLOAD_DIR
from app.utils.file_utils import calculate_checksum
from app.services.database import engine
from app.services.memory_service import (
    get_all_sessions,
    get_session_history,
    semantic_search_session,
    delete_session_memory,
)
from app.services.agentic_rag_service import run_agent

router = APIRouter(prefix="/rag", tags=["RAG"])


# =========================
# UPLOAD
# =========================


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    session_id: str,
    file: UploadFile = File(...),
):
    # -------------------------
    # READ + CHECKSUM
    # -------------------------

    file_bytes = await file.read()
    checksum = calculate_checksum(file_bytes)

    # -------------------------
    # CHECKSUM LOOKUP (Step 2.2 / 2.3)
    # -------------------------

    with Session(engine) as db:
        existing_doc = db.exec(
            select(Document).where(Document.checksum == checksum)
        ).first()

        if existing_doc:
            # document already exists — just create the session mapping if missing
            existing_mapping = db.exec(
                select(SessionDocument).where(
                    SessionDocument.session_id == session_id,
                    SessionDocument.document_id == existing_doc.id,
                )
            ).first()

            if not existing_mapping:
                db.add(
                    SessionDocument(
                        session_id=session_id,
                        document_id=existing_doc.id,
                    )
                )
                db.commit()

            return {
                "message": "File already indexed, linked to session",
                "status": existing_doc.status,
                "original_filename": existing_doc.original_filename,
                "stored_filename": existing_doc.stored_filename,
                "document_id": existing_doc.id,
                "reused": True,
            }

    # -------------------------
    # NEW DOCUMENT — build stored filename
    # -------------------------

    file_size = len(file_bytes)
    base_name = os.path.splitext(file.filename)[0]
    stored_filename = f"{base_name}_{checksum[:16]}_{file_size}.pdf"

    # -------------------------
    # SAVE FILE LOCALLY
    # -------------------------

    file_path = os.path.join(UPLOAD_DIR, stored_filename)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # -------------------------
    # CREATE DOCUMENT ROW
    # -------------------------

    with Session(engine) as db:
        doc = Document(
            checksum=checksum,
            original_filename=file.filename,
            stored_filename=stored_filename,
            status="processing",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        db.add(
            SessionDocument(
                session_id=session_id,
                document_id=doc.id,
            )
        )
        db.commit()

        document_id = doc.id

    # -------------------------
    # BACKGROUND INDEXING
    # -------------------------

    background_tasks.add_task(build_vector_database, document_id, stored_filename)

    return {
        "message": "File uploaded, indexing started",
        "status": "processing",
        "original_filename": file.filename,
        "stored_filename": stored_filename,
        "document_id": document_id,
        "reused": False,
    }


# =========================
# DOCUMENTS
# =========================


@router.get("/documents")
async def documents():
    return {"documents": list_ready_documents()}


# =========================
# ASK
# =========================


@router.post("/ask")
async def ask_pdf(payload: AskDocumentRequest, background_tasks: BackgroundTasks):
    answer = await ask_document(
        session_id=payload.session_id,
        question=payload.question,
        background_tasks=background_tasks,
        document_ids=payload.document_ids,
    )
    return {"answer": answer}


@router.post("/ask/langchain")
async def ask_pdf_langchain(
    payload: AskDocumentRequest,
    background_tasks: BackgroundTasks,
    stream: bool = False,
):
    result = await ask_document_langchain(
        session_id=payload.session_id,
        question=payload.question,
        background_tasks=background_tasks,
        stream=stream,
        document_ids=payload.document_ids,  # None = search all session docs
    )

    if stream:
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

    return {"answer": result}


@router.post("/ask/agent")
async def ask_agent(payload: AgentAskRequest):
    result = await run_agent(
        session_id=payload.session_id,
        question=payload.question,
        document_ids=payload.document_ids,
        stream=payload.stream,
    )

    if payload.stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return {"answer": result}


# =========================
# SESSIONS
# =========================


@router.get("/sessions")
async def sessions():
    return {"sessions": get_all_sessions()}


@router.get("/sessions/{session_id}")
async def session_history(session_id: str, page: int = 1, page_size: int = 20):
    history = get_session_history(session_id, page, page_size)
    return {"session_id": session_id, "history": history}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(delete_session_full, session_id)
    return {"message": "Session deletion started."}


@router.post("/sessions/{session_id}/search")
async def search_session_memory(session_id: str, payload: SearchMemoryRequest):
    results = semantic_search_session(session_id, payload.query)
    return {"results": results}


@router.post("/sessions")
async def create_new_session(title: str | None = None):
    return create_session(title=title)


@router.get("/sessions/{session_id}/documents")
async def session_documents(session_id: str):
    return {"documents": get_session_documents(session_id)}


@router.delete("/sessions/{session_id}/documents/{document_id}")
async def remove_document_from_session(session_id: str, document_id: str):
    return unlink_session_document(session_id, document_id)


@router.post("/admin/cleanup")
async def trigger_orphan_cleanup(background_tasks: BackgroundTasks):
    """Manual trigger for orphan cleanup — useful for admin/maintenance."""
    background_tasks.add_task(cleanup_orphan_documents)
    return {"message": "Orphan cleanup started."}
