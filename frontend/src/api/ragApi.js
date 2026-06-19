import axios from 'axios'
import { getEndpointConfig, isValidStrictness, DEFAULT_STRICTNESS } from '../config/endpoints'

const api = axios.create({ baseURL: '' })

// ── Documents ─────────────────────────────────────────────

export const getDocuments = (signal) =>
  api.get('/rag/documents', { signal }).then(r => r.data.documents)

export const uploadDocument = (file, sessionId, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  return api.post(`/rag/upload?session_id=${encodeURIComponent(sessionId)}`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress?.(Math.round(e.loaded * 100 / e.total)),
  }).then(r => r.data)
}

// ── Sessions ──────────────────────────────────────────────

export const getSessions = (signal) =>
  api.get('/rag/sessions', { signal }).then(r => r.data.sessions)

export const createSession = (title) =>
  api.post('/rag/sessions', null, { params: { title } }).then(r => r.data)

export const getSessionHistory = (sessionId, page = 1, pageSize = 50, signal) =>
  api.get(`/rag/sessions/${sessionId}`, { params: { page, page_size: pageSize }, signal })
    .then(r => r.data.history?.items || [])

export const deleteSession = (sessionId) =>
  api.delete(`/rag/sessions/${sessionId}`).then(r => r.data)

export const getSessionDocuments = (sessionId, signal) =>
  api.get(`/rag/sessions/${sessionId}/documents`, { signal }).then(r => r.data.documents)

export const unlinkDocument = (sessionId, documentId) =>
  api.delete(`/rag/sessions/${sessionId}/documents/${documentId}`).then(r => r.data)

// ── Ask — polymorphic, driven by the ENDPOINTS registry ────
//
// Replaces the old per-endpoint askDocument/askDocumentStream pair. One
// function handles all 6 endpoints; endpoint-specific quirks (legacy query
// param vs new-format body field, which fields are even applicable) live
// here, isolated from useChat.js, which only ever branches on stream: true/false.

/**
 * @param {string} endpointId - one of the ids in src/config/endpoints.js
 * @param {object} params
 * @param {string} params.sessionId
 * @param {string} params.question
 * @param {string[]|null} [params.documentIds] - omitted entirely if falsy/empty
 * @param {boolean} [params.stream]
 * @param {boolean} [params.useWeb] - ignored if the endpoint doesn't support it
 * @param {string|null} [params.strictness] - ignored if the endpoint doesn't
 *   support it; falls back to DEFAULT_STRICTNESS if invalid/omitted
 * @returns {Promise<object|Response>} parsed JSON for a non-streaming call,
 *   or the raw fetch Response (for createSSEStreamReader) for a streaming call
 */
export async function askEndpoint(endpointId, {
  sessionId,
  question,
  documentIds = null,
  stream = false,
  useWeb = false,
  strictness = null,
} = {}) {
  const config = getEndpointConfig(endpointId)
  if (!config) {
    throw new Error(`askEndpoint: unknown endpoint id "${endpointId}"`)
  }

  // An endpoint that doesn't support streaming can never stream, no matter
  // what the caller's streaming toggle says — this protects against the
  // global toggle being on while "Simple" (/rag/ask) is selected.
  const wantsStream = config.supportsStream && !!stream

  // Build the request body with only the fields this endpoint actually
  // accepts (per the matrix in the integration plan). Never send a key the
  // endpoint doesn't understand, and never send `undefined`.
  const body = { session_id: sessionId, question }
  if (documentIds && documentIds.length) body.document_ids = documentIds

  // legacySSE (langchain) signals streaming via a `?stream=` query param,
  // not a body field — /rag/ask has no streaming concept at all either way.
  if (config.id !== 'ask' && !config.legacySSE) {
    body.stream = wantsStream
  }
  if (config.supportsWeb) {
    body.use_web = !!useWeb
  }
  if (config.supportsStrictness) {
    body.strictness = isValidStrictness(strictness) ? strictness : DEFAULT_STRICTNESS
  }

  if (wantsStream) {
    const url = config.legacySSE ? `${config.path}?stream=true` : config.path
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  }

  return api.post(config.path, body).then(r => r.data)
}
