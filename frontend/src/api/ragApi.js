import axios from 'axios'

const api = axios.create({ baseURL: '' })

// ── Documents ────────────────────────────────────────────
export const getDocuments = () =>
  api.get('/rag/documents').then(r => r.data.documents)


export const uploadDocument = (file, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/rag/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress?.(Math.round(e.loaded * 100 / e.total)),
  }).then(r => r.data)
}

// ── Sessions ─────────────────────────────────────────────
export const getSessions = () =>
  api.get('/rag/sessions').then(r => r.data.sessions)

export const getSessionHistory = (sessionId, page = 1, pageSize = 50) =>
  api.get(`/rag/sessions/${sessionId}`, { params: { page, page_size: pageSize } }).then(r => r.data.history.items || [])

export const deleteSession = (sessionId) =>
  api.delete(`/rag/sessions/${sessionId}`).then(r => r.data)

// ── Ask — JSON ────────────────────────────────────────────
export const askDocument = (filename, question, sessionId) =>
  api.post('/rag/ask/langchain', { filename, question, session_id: sessionId })
    .then(r => r.data)

// ── Ask — Stream ──────────────────────────────────────────
export const askDocumentStream = (filenames, question, sessionId) =>
  fetch('/rag/ask/langchain?stream=true', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filenames, question, session_id: sessionId }),
  })
