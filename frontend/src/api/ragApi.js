import axios from 'axios'

const api = axios.create({ baseURL: '' })

// ── Documents ────────────────────────────────────────────
export const getDocuments = () =>
  api.get('/rag/documents').then(r => r.data.documents)

// ── Sessions ─────────────────────────────────────────────
export const getSessions = () =>
  api.get('/rag/sessions').then(r => r.data.sessions)

export const getSessionHistory = (sessionId, page = 1, pageSize = 50) =>
  api.get(`/rag/sessions/${sessionId}`, { params: { page, page_size: pageSize } }).then(r => r.data.history.items || [])

export const deleteSession = (sessionId) =>
  api.delete(`/rag/sessions/${sessionId}`).then(r => r.data)

// ── Ask — JSON ────────────────────────────────────────────
export const askDocument = (filenames, question, sessionId) =>
  api.post('/rag/ask/langchain', { filenames, question, session_id: sessionId })
    .then(r => r.data)

// ── Ask — Stream ──────────────────────────────────────────
// Returns a fetch response for manual stream reading
// (axios doesn't support streaming natively)
export const askDocumentStream = (filenames, question, sessionId) =>
  fetch('/rag/ask/langchain?stream=true', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filenames, question, session_id: sessionId }),
  })
