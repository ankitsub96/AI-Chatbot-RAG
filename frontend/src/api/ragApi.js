import axios from 'axios'

const api = axios.create({ baseURL: '' })

// ── Documents ─────────────────────────────────────────────

export const getDocuments = () =>
  api.get('/rag/documents').then(r => r.data.documents)

export const uploadDocument = (file, sessionId, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  return api.post(`/rag/upload?session_id=${encodeURIComponent(sessionId)}`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress?.(Math.round(e.loaded * 100 / e.total)),
  }).then(r => r.data)
}

// ── Sessions ──────────────────────────────────────────────

export const getSessions = () =>
  api.get('/rag/sessions').then(r => r.data.sessions)

export const createSession = (title) =>
  api.post('/rag/sessions', null, { params: { title } }).then(r => r.data)

export const getSessionHistory = (sessionId, page = 1, pageSize = 50) =>
  api.get(`/rag/sessions/${sessionId}`, { params: { page, page_size: pageSize } })
    .then(r => r.data.history?.items || [])

export const deleteSession = (sessionId) =>
  api.delete(`/rag/sessions/${sessionId}`).then(r => r.data)

export const getSessionDocuments = (sessionId) =>
  api.get(`/rag/sessions/${sessionId}/documents`).then(r => r.data.documents)

export const unlinkDocument = (sessionId, documentId) =>
  api.delete(`/rag/sessions/${sessionId}/documents/${documentId}`).then(r => r.data)

// ── Ask — Stream ──────────────────────────────────────────

export const askDocumentStream = (sessionId, question, documentIds = null) =>
  fetch('/rag/ask/langchain?stream=true', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      question,
      ...(documentIds ? { document_ids: documentIds } : {}),
    }),
  })

// ── Ask — JSON ────────────────────────────────────────────

export const askDocument = (sessionId, question, documentIds = null) =>
  api.post('/rag/ask/langchain', {
    session_id: sessionId,
    question,
    ...(documentIds ? { document_ids: documentIds } : {}),
  }).then(r => r.data)
