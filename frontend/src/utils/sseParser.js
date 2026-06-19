// Single shared SSE (Server-Sent Events) parsing layer for both wire formats
// used by the RAG endpoints:
//
//   Legacy (/rag/ask/langchain):
//     data: {"thinking": "stage_name", "message": "..."}
//     data: {"token": "..."}
//     data: {"done": true}
//
//   New (/rag/ask/agent, /rag/react/ask, /rag/planner/ask, /rag/research/ask):
//     event: thinking
//     data: {"type": "thinking", "stage": "...", "message": "...", "data": {...}}
//
//     event: response
//     data: {"type": "response", "token": "..."}
//
//     event: done
//     data: {"type": "done"}
//
// Both are normalized into one internal shape before reaching any
// hook/component:
//   { kind: 'thinking', stage: string, message: string, data: object }
//   { kind: 'token', token: string }
//   { kind: 'done' }
//
// This is the ONLY place in the app that touches response.body.getReader().

/**
 * Normalize a legacy-format data payload (no `event:` line — the shape of
 * the event is inferred from which key is present on the object).
 * @param {object} dataObj
 * @returns {{kind:'thinking',stage:string,message:string,data:object}|{kind:'token',token:string}|{kind:'done'}|null}
 */
export function normalizeLegacy(dataObj) {
  if (!dataObj || typeof dataObj !== 'object') return null

  if (dataObj.thinking !== undefined) {
    return {
      kind: 'thinking',
      stage: dataObj.thinking,
      message: dataObj.message || '',
      data: dataObj.data || {},
    }
  }

  if (dataObj.token !== undefined) {
    return { kind: 'token', token: dataObj.token }
  }

  if (dataObj.done) {
    return { kind: 'done' }
  }

  return null
}

/**
 * Normalize a new-format data payload. `eventType` comes from the most
 * recent `event:` line seen by the caller; falls back to `dataObj.type` if
 * the caller didn't track one (defensive — the server sends both
 * redundantly, so this keeps things working even if a future endpoint omits
 * the `event:` line).
 * @param {string|null} eventType
 * @param {object} dataObj
 */
export function normalizeNew(eventType, dataObj) {
  if (!dataObj || typeof dataObj !== 'object') return null
  const kind = eventType || dataObj.type

  switch (kind) {
    case 'thinking':
      return {
        kind: 'thinking',
        stage: dataObj.stage || 'thinking',
        message: dataObj.message || '',
        data: dataObj.data || {},
      }

    case 'message':
      if (dataObj.thinking !== undefined) {
        return {
          kind: 'thinking',
          stage: dataObj.thinking,
          message: dataObj.message || '',
          data: dataObj.data || {},
        }
      }
      return null

    case 'response':
      return { kind: 'token', token: dataObj.token || '' }

    case 'done':
      return { kind: 'done' }

    default:
      return null // unknown/forward-compatible event type — ignore, don't throw
  }
}

/**
 * Parse a single raw SSE `data:` line into a normalized event.
 * Pure function — no I/O, no shared state — so it's trivially unit-testable
 * on its own, independent of any real network stream.
 *
 * @param {string} rawLine - one line of the SSE stream, e.g. `data: {"token":"hi"}`
 * @param {string|null} currentEventType - the most recent `event:` value seen
 *   (null/undefined for the legacy format, which never sends `event:` lines)
 * @returns {object|null} a NormalizedEvent, or null if the line isn't a data
 *   line, isn't valid JSON (e.g. a chunk boundary cut a JSON object in half),
 *   or doesn't map to a known event kind.
 */
export function parseSSEChunk(rawLine, currentEventType) {
  const line = (rawLine || '').trim()
  if (!line.startsWith('data:')) return null

  const raw = line.slice(5).trim()
  if (!raw) return null

  let parsed
  try {
    parsed = JSON.parse(raw)
  } catch {
    return null // partial/invalid JSON — createSSEStreamReader's line
    // buffering should prevent this in practice, but stay safe
  }

  return currentEventType
    ? normalizeNew(currentEventType, parsed)
    : normalizeLegacy(parsed)
}

/**
 * Consume a fetch Response's streaming body, normalizing both legacy and
 * new SSE wire formats, calling `onEvent(normalized)` for every event as it
 * arrives. Buffers partial lines across chunk boundaries (a network chunk
 * can split a JSON payload mid-object — without this, that event would
 * silently get dropped).
 *
 * @param {Response} response
 * @param {(event: object) => void} onEvent
 * @returns {Promise<void>} resolves once the stream truly ends. A `done`
 *   event from the server is not guaranteed, so callers needing to know the
 *   stream is finished should await this promise rather than rely solely on
 *   receiving `{ kind: 'done' }`.
 */
export async function createSSEStreamReader(response, onEvent) {
  if (!response?.body) {
    throw new Error('createSSEStreamReader: response has no readable body')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = '' // trailing partial line carried across read() calls
  let currentEventType = null

  const processLine = (rawLine) => {
    const line = rawLine.trim()

    if (!line) {
      // blank line = SSE message boundary; event type doesn't carry over
      currentEventType = null
      return
    }

    if (line.startsWith(':')) return // SSE comment line, ignore

    if (line.startsWith('event:')) {
      currentEventType = line.slice(6).trim()
      return
    }

    if (!line.startsWith('data:')) return // id:, retry:, etc. — unused here

    const normalized = parseSSEChunk(line, currentEventType)
    if (normalized) onEvent(normalized)
  }

  while (true) {
    const { done, value } = await reader.read()

    if (value) {
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? '' // keep the trailing partial line for next time
      for (const line of lines) processLine(line)
    }

    if (done) {
      // flush the decoder + whatever's left in the buffer — covers a stream
      // that ends without a trailing newline
      buffer += decoder.decode()
      if (buffer) processLine(buffer)
      break
    }
  }
}
